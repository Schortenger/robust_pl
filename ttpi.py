'''
    Copyright (c) 2022 XYZ
    Written by XYZ,

    Paper: "TTPI: Generalized Policy Iteration using Tensor Approximation" 
 
    This class contains the pytorch implementation of the whole pipeline of TTPI(Generalized 
                                                                                Policy Iteration using Tensor Train):


     - Input:
        - reward: reward function ,
        - gamma: discount factor in range (0,1). Default 0.99
            Note: larger the gamma the larger the horizon for decision making but slower the convergence 
        - domain_state: discretization of the state (and possibly includes the parameters)
        - domain_action: the discretization of the control/action space 
        (each is a list containing a 1-D tensor or discretization points of each axis)
        - forward_model: system dynamics (given (s,a) find the next state s'=f(s,a))

    TTPI exploits the following properties of TT format:
        - cross-approximation algorithm: used to find the value function (V(s)), Advantage-function (A(s,a)) 
        - batch optimization for optimal actions given a state: used to choose a good 
        action given the state (i.e. a suboptimal version of: armax A(a|s))          

'''
import torch
import tntorch as tnt

import copy
import warnings
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=6)
import sys
from sys import getsizeof as gs
import pdb
import time
import math 

from tt_utils import *

class TTPI:
    def __init__(self, 
        domain_state, domain_action, 
        reward, forward_model, dt=0.01,
        interpolated_state=None, 
        gamma=0.99,
        q_lb=1e1, q_ub=1e2,
        n_steps=1,n_step_a=1,
        max_batch_v=10**4, max_batch_a=10**5, 
        nswp_v=20, nswp_a=20, 
        rmax_v=100, rmax_a=100, 
        eps_round_v=1e-3,eps_round_a=1e-3,
        eps_cross_v=1e-4,eps_cross_a=1e-4,

        kickrank_v=2, kickrank_a=5, 
        n_samples=100, action_random=False,
        alpha=0.9, beta=1.0,
        normalize_reward=False, 
        verbose=False, device="cpu"):

        self.dt = dt # time-step for discrete control (should be same as used in forward-model)
        
        self.device = device
        
        # Discretization of each axis 
        self.domain_state = [x.to(self.device) for x in domain_state] # a list of  1-D torch-tensors containing the discretization points along each axis/mode 
        self.domain_action = [x.to(self.device) for x in domain_action] # a list of  1-D torch-tensors containing the discretization points along each axis/mode      
        self.domain_state_action = self.domain_state + self.domain_action

        # Find the lower bound and step size of the discretization
        self.min_state_action = torch.tensor([x[0].to(self.device) for x in self.domain_state_action]).to(device)
        self.dh_state_action = torch.tensor([(x[1]-x[0]).abs().to(self.device) for x in self.domain_state_action]).to(device)
        self.dh_state = torch.tensor([(x[1]-x[0]).abs().to(self.device) for x in self.domain_state]).to(device)        

        # Number of discretization points
        self.d_state = torch.tensor([len(x) for x in domain_state]).to(device) # number of discretization points along each axis/mode of statespace
        self.d_action = torch.tensor([len(x) for x in domain_action]).to(device) # number of discretization points along each axis/mode of statespace        
        self.d_state_action = torch.concat((self.d_state,self.d_action)).view(-1)
        self.num_states = torch.prod(self.d_state).item()
        
        # Dimension of the state space and the state-action space
        self.dim_state = len(self.d_state)
        self.dim_action = len(self.d_action)
        self.dim_state_action = self.dim_state + self.dim_action

        # Reward function
        self.reward = reward # reward function r(s,a)
        self.gamma = gamma # discount factor in range (0,1)
        
        # System forward simulation
        self.forward_model = forward_model # given (s,a) find the next state
        
        # tt-cross params
        self.max_batch_v =max_batch_v+2 # maximum batch size for cross-approximation (to avoid memory overflow)
        self.max_batch_a= max_batch_a+2
        
        self.nswp_v=nswp_v
        self.nswp_a=nswp_a
        self.rmax_a=rmax_a
        self.rmax_v=rmax_v
        self.eps_round_v=eps_round_v
        self.eps_round_a=eps_round_a
        self.eps_cross_v=eps_cross_v
        self.eps_cross_a=eps_cross_a
        self.kickrank_a=kickrank_a
        self.kickrank_v=kickrank_v
        self.verbose=verbose
        
        # The following are lower and upper bounds used to shift the q-function (because it needs to be non-negative to optimize)
        self.q_lb = torch.tensor([q_lb])#.to(self.device)
        self.q_ub = torch.tensor([q_ub])#.to(self.device)
        self.n_steps=n_steps
        self.n_step_a = n_step_a

        # For sampling from TT-distribution (i.e. the conditional sampling from Q-model Q(s|a))
        self.n_samples=n_samples
        self.alpha=alpha # prioritized sampling if stochastic method is used for tt-based optimization
        self.beta = beta # used to scale the advantage function (not that important, it will be removed in later implementation )
        self.action_random = action_random
        if self.action_random:#random slection from action space
            self.policy = self.policy_random
        else:
            self.policy = self.policy_ttgo

        if interpolated_state is None: # all are interpolated
            self.interpolated_state = torch.tensor([True]*self.dim_state).to(self.device)
        else:
            self.interpolated_state = interpolated_state
        
        self.normalize_reward = normalize_reward
            

    @torch.no_grad()
    def train(self, 
                n_iter_max=1000, n_iter_v=1,
                resume=False, callback=None, callback_freq=20,
                verbose=False,
                file_name='ttdp_model'):
        print("#############################################################################")
        print("Learning begins")
        print("#############################################################################")
       
        self.verbose = verbose
        torch.cuda.empty_cache()
        self.rand_state = torch.random.get_rng_state()
        if not resume:
            reward_tt = self.get_reward_model()
            self.reward_tt = reward_tt.to(self.device)
            if not self.normalize_reward:
                self.reward_max = 1.0
            else:
                self.reward_max = torch.abs(self.get_max_a(self.reward_tt)[0])
            self.reward_normalized_tt = self.reward_tt*(1/self.reward_max)
            self.reward_normalized_tt.round(1e-9)
            self.reward_normalized_tt = self.reward_normalized_tt.to(self.device)
            print("Initialize policy (q-fcn) by random initialization of value-function....")
            self.a_model = self.reward_normalized_tt.clone().to(self.device)#self.policy_improve().to(self.device)
            self.v_model = (0*tnt.rand(self.d_state,ranks_tt=1).tt()).to(self.device)
            self.policy_model = self.normalize_tt_a(self.a_model.clone()).to(self.device)
            iter=0
        else:
            print("Initializing value-function from the previous run....")
            model = torch.load(file_name+'.pt')
            self.reward_tt = model['reward_tt'].to(self.device)
            self.reward_max = self.get_max_a(self.reward_tt)[0]
            if not self.normalize_reward:
                self.reward_max = 1.0
            else:
                self.reward_max = torch.abs(self.get_max_a(self.reward_tt)[0])
            self.reward_normalized_tt = self.reward_tt*(1/self.reward_max)
            self.reward_normalized_tt.round(1e-9)
            self.reward_normalized_tt = self.reward_normalized_tt.to(self.device)  
            self.a_model = model['a_model'].to(self.device)
            self.v_model = model['v_model'].to(self.device)
            self.policy_model = model['policy_model'].to(self.device)
            iter = model['iter']


        self.policy_model_cores = self.policy_model.tt().cores[:]
        # print("Rank of reward-model: ", (self.reward_normalized_tt.ranks_tt))
        self.train_data = { "v_norm":[],  "v_mean":[], "v_rank":[],
                            "a_norm":[],  "a_mean":[], "a_rank":[],
                            "q_norm":[],  "q_mean":[], "q_rank":[], 
                            "p_norm":[],  "p_mean":[], "p_rank":[], 
                            "final_reward":[], 
                            "cum_reward":[],
                            "dv_norm":[]}
                        
                       
                        
        if not (callback is None):
            reward, cum_reward = callback(self,callback_count=0)
            self.train_data['final_reward']+=[reward.cpu()]
            self.train_data['cum_reward']+=[cum_reward.cpu()]

        for i in range(iter,n_iter_max):
            print("Memory: ", torch.cuda.memory_allocated()*1e-9,torch.cuda.memory_reserved()*1e-9)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Policy Iteration {}/{}".format(i+1,n_iter_max))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            self.rand_state = torch.random.get_rng_state()


            t1=time.time()

            self.PI_update(n_iter_v=n_iter_v) # updates self.a_model and self.v_model
            self.policy_model = self.normalize_tt_a(self.a_model.clone())                   
        
            self.policy_model_cores = self.policy_model.tt().cores[:]
            
            t2=time.time()

            self.log_data() 

            torch.cuda.empty_cache()

            print("--------------------------------------------")
            print("Time taken:{}".format(t2-t1))
            print("--------------------------------------------")

            if (not (callback is None)) and ((i+1)%callback_freq==0):
                print("---------------Intermeditate Test-------------------------")
                reward, cum_reward = callback(self,callback_count=i)
                self.train_data['final_reward']+=[reward.cpu()]
                self.train_data['cum_reward']+=[cum_reward.cpu()]
                self.plt_training_stat()
                self.save_model(file_name,i)
                if cum_reward > torch.max(torch.tensor(self.train_data['cum_reward'])):
                    print("Improved policy saved, iteration={}".format(i))
                    self.save_model(file_name+'_best',i)
    
    def PI_update(self, n_iter_v=1):
        '''
           Do n_iter_v steps of Policy evaluation and update a_model 
        '''
        print("Number of value updates: ", n_iter_v )
        v_model = self.v_model.clone()
        for i in range(n_iter_v):
            v_model = self.compute_value_model(v_model).to(self.device) # updates value model
            if n_iter_v>1:
                v_model.round_tt(self.eps_round_v)
                print("Iteration:{}, Rank of v-model:{}".format(i,v_model.ranks_tt))
        self.v_model = v_model.clone()
        self.a_model = self.compute_advantage_model_from_value().to(self.device)

    def VI_update(self):
        '''
            Update v_model and a_model 
        '''
        self.v_model = self.compute_value_model(self.v_model).to(self.device) # updates value model
        self.a_model = self.compute_advantage_model_from_value().to(self.device)
        # self.a_model.round_tt(self.eps_round_a)
    

    def extend_model(self, tt_model):
        ''' 
        Given a tt_model function of state (s) 
        returns an extende model tt_model_ext function of (state,action) s
        tt_model_ext(s,a) = tt_model(s)
        '''
        base_cores = tt_model.tt().cores[:]
        r_ = base_cores[self.dim_state-1].shape[-1]
        id_action = torch.eye(r_)[:,None,:]
        cores = base_cores[:] + [id_action.expand(-1, self.d_action[i],-1) for i in range(self.dim_action)]
        tt_model_ext = tnt.Tensor(cores).to(self.device)
        # maybe round it?
        return tt_model_ext

    def get_reward_model(self):
        def reward_fcn(state_action):
            state = state_action[:,:self.dim_state].view(-1,self.dim_state)
            action = state_action[:,self.dim_state:].view(-1,self.dim_action)
            return self.reward(state,action)
        print("Computing reward function in TT format for normalization")
        
        reward_tt = self.cross_approximate(fcn=reward_fcn, eps=self.eps_cross_a*1e-1, 
                                    max_batch=self.max_batch_a, domain=self.domain_state_action, 
                                    nswp=self.nswp_a, rmax=self.rmax_a, kickrank=self.kickrank_a, 
                                    verbose=True)
        reward_tt.round_tt(eps=self.eps_round_a*1e-1)
        print("Rank of reward: ", reward_tt.ranks_tt)
        return reward_tt
    
    def reward_normalized(self,state,action):
        '''
        Normalize the reward to be approximately in range (-self.beta,self.beta) if self.normalize is True
        '''
        return self.reward(state,action)*self.beta/(1e-6+self.reward_max)
    
    
    def normalize_tt_v(self,tt_model):
        return self.normalize_tt(tt_model,self.domain_state)


    def normalize_tt_a(self,tt_model):
        return self.normalize_tt(tt_model,self.domain_state_action)
     

    def normalize_tt(self,tt_model, domain):
        '''
        Monotonic transformation of elements of tt_model 
        so that the values are in between self.q_lb and self.q_ub
        If auto_boundm, then just shift the elements to positive numbers
        without any scaling (this is recommended to avoid numerical issues)
        '''
        tt_model_n = normalize_tt(tt_model.clone(), domain=domain,
                     lb=self.q_lb, ub=self.q_ub,
                     auto_bound=True, canonicalize=True,
                     device=self.device) # from tt_utils
        return tt_model_n
        
    def compute_value_model(self,v_model):
        ''' 
        Find the value function corresponding to the current Q-fcn using TT-Cross 
        '''
        v_model = self.apply_bellman(v_model) # one step of value iteration for the current policy (Q-fcn) 
        return v_model
    
    def compute_ref_model(self,a_model):
    
        normalized_a_model = self.normalize_tt_a(a_model).to(self.device)
        def a_ref_fcn(state):
            state_action = self.action_tt(tt_cores=normalized_a_model.tt().cores[:], 
                                                    domain=self.domain_state_action,state=state).to(self.device)
            return self.get_value_a(a_model,state_action[:,0,:]).view(-1)
        
        a_ref_model = self.cross_approximate(fcn=a_ref_fcn, 
                    eps=self.eps_cross_a, max_batch=self.max_batch_a, 
                    domain=self.domain_state, nswp=self.nswp_a, 
                    rmax=self.rmax_a, kickrank=self.kickrank_a, verbose=self.verbose) 
        return a_ref_model.to(self.device)           


    def compute_target_a_model(self, direct_method=True):
        ''' 
        Find the target-Q-function (Target to Q-learning):
        W(s,a) = R(s,a) + gamma*max_a' Q(f(s,a),a') 
        '''
        if direct_method:
            target_a_model = self.cross_approximate(fcn=self.get_target_a, 
                    eps=self.eps_cross_a, max_batch=self.max_batch_a, 
                    domain=self.domain_state_action, nswp=self.nswp_a, 
                    rmax=self.rmax_a, kickrank=self.kickrank_a, verbose=self.verbose)
        else:
            def get_u(state_action):
                '''
                    Given state_action=(s,a), get U(s,a) = max_a'(Q(f(s,a),a'))
                    input: state-action pair, batch_size x (dim_state+dim_action)
                '''

                state = state_action[:,:self.dim_state].view(-1,self.dim_state)
                action = state_action[:,self.dim_state:].view(-1,self.dim_action)
                next_state = self.forward_model(state,action) # batch_size x dim_state
                next_action = self.policy(next_state)
                next_state_action = torch.cat((next_state,next_action),dim=-1)
                u_values = self.get_value_a(self.q_model,next_state_action) # batch_size x 1
                return u_values

            u_model = self.cross_approximate(fcn=get_u, 
                    eps=self.eps_cross_v, max_batch=self.max_batch_v, 
                    domain=self.domain_state_action, nswp=self.nswp_v, 
                    rmax=self.rmax_v, kickrank=self.kickrank_v, verbose=self.verbose)
            target_a_model = self.reward_normalized_tt + self.gamma*u_model
            target_a_model.round_tt(self.eps_round_a)
        
        return target_a_model.to(self.device)
        

    def compute_a_model_from_value(self):
        ''' 
            TT-Cross Approximation to find the Q-function 
            Q(s,a)=R(s,a)+gamma*V(f(s,a).
        '''
        q_model = self.cross_approximate(fcn=self.get_a_from_value, 
                eps=self.eps_cross_a, max_batch=self.max_batch_a, 
                domain=self.domain_state_action, nswp=self.nswp_a, 
                rmax=self.rmax_a, kickrank=self.kickrank_a, verbose=self.verbose)
        return q_model.to(self.device)
    
    def compute_advantage_model_from_a(self):
        ''' 
            TT-Cross Approximation to find the Advantage-function 
                A(s,a)=R(s,a)+gamma*max_a'Q(f(s,a)),a') - max_a Q(a,a, compute directly in TT-Cross
        '''
        print('.....................................................')
        print("Computing Advantage Fcn")
        print('.....................................................')  


        a_model = self.cross_approximate(self.get_advantage_from_a, 
                    eps=self.eps_cross_a, max_batch=self.max_batch_a, 
                    domain=self.domain_state_action, nswp=self.nswp_a, 
                    rmax=self.rmax_a, kickrank=self.kickrank_a, verbose=self.verbose)

        return a_model.to(self.device)



    def compute_advantage_model_from_value(self, direct_method=True):
        ''' 
            TT-Cross Approximation to find the Advantage-function 
            direct_method:
                A(s,a)=R(s,a)+gamma*V(f(s,a))-V(s)), compute directly in TT-Cross
            indirect:
                First compute V(s,a) = V(f(s,a)) using TT-Cross
                Then: 
                A(s,a) = R(s,a) + gamma*V(s,a) - V(s). Each term is in TT and hence use TT arithmetics
        '''
        print('.....................................................')
        print("Computing Advantage Fcn")
        print('.....................................................')  


        if direct_method:
            a_model = self.cross_approximate(fcn=self.get_advantage_from_value, 
                    eps=self.eps_cross_a, max_batch=self.max_batch_a, 
                    domain=self.domain_state_action, nswp=self.nswp_a, 
                    rmax=self.rmax_a, kickrank=self.kickrank_a, verbose=self.verbose)
        else:
            dv_sa_model = self.compute_dv_sa_model() # get V(s,a) = V(f(s,a)) using TT-Cross
            a_model = self.reward_normalized_tt.to(self.device)*self.dt + self.gamma*dv_sa_model.to(self.device)
            a_model.round_tt(self.eps_cross_a)

        return a_model.to(self.device)
    

    def apply_bellman(self,v_model):
        ''' 
            TT-Cross Approximation to update the value function based on bellman equation.
            eps: precentage change in the norm of tt per iteration of tt-cross
        '''
        def bellman_operator(state):
            '''
            The output of bellman operator for the current policy at the given states.
            i.e. it returns: R(s,a) + gamma*V(s') where s' = f(s,a), a = policy(s)
            input: state, batch_size x dim_state
            output: value of bellman-operator at the given states, batch_size
            '''
            cum_reward=0.
            for i in range(self.n_steps):
                action = self.policy(state) # get the action given the state, batch_size x dim_action
                cum_reward += (self.gamma**i)*self.reward_normalized(state,action) 
                state = self.forward_model(state,action) # get the next state given current state and action, batch_size x dim_state                
            cum_reward*=self.dt
            cum_reward += (self.gamma**(self.n_steps))*self.get_value_v(v_model,state) # batch_size x 1
            
            return cum_reward

        v_model = self.cross_approximate(fcn=bellman_operator,max_batch=self.max_batch_v,
                                                eps=self.eps_cross_v, 
                                                domain=self.domain_state, rmax=self.rmax_v, 
                                                nswp=self.nswp_v, verbose=self.verbose,
                                                kickrank=self.kickrank_v)
        return v_model


    def get_a_from_value(self,state_action):
        '''
            Given state_action=(s,a), get the output of advantage-function using the current approximation of value fcn
            input: state-action pair, batch_size x (dim_state+dim_action)
        '''

        state = state_action[:,:self.dim_state].view(-1,self.dim_state)
        action = state_action[:,self.dim_state:].view(-1,self.dim_action)
        next_state = self.forward_model(state,action) # batch_size x dim_state
        q_values = self.reward_normalized(state,action) + self.gamma*self.get_value_v(self.v_model,next_state) # batch_size x 1
        return q_values

    def get_a_next_sa(self,state_action):
        '''
            Given state_action=(s,a), get the output of advantage-function A(f(s,a),policy(f(s,a)))
             using the current approximation of value fcn
            input: state-action pair, batch_size x (dim_state+dim_action)
        '''

        state = state_action[:,:self.dim_state].view(-1,self.dim_state)
        action = state_action[:,self.dim_state:].view(-1,self.dim_action)
        next_state = self.forward_model(state,action) # batch_size x dim_state
        next_action = self.policy(next_state)
        next_state_action = torch.cat((next_state,next_action),dim=-1)
        target_a_values = self.get_value_a(self.a_model,next_state_action) # batch_size x 1
        return target_a_values
    
    def compute_a_next_sa(self):
        a_next_sa = self.cross_approximate(fcn=self.get_a_next_sa, eps=self.eps_round_a, 
                         max_batch=self.max_batch_a, 
                        domain=self.domain_state_action, nswp=self.nswp_a, 
                        rmax=self.rmax_a, 
                        kickrank=self.kickrank_a, verbose=self.verbose)
        a_next_sa.round_tt(self.eps_cross_a)
        return a_next_sa.to(self.device)        



    def get_target_a(self,state_action):
        '''
            Given state_action=(s,a), get the output of advantage-function using the current approximation of value fcn
            input: state-action pair, batch_size x (dim_state+dim_action)
        '''

        state = state_action[:,:self.dim_state].view(-1,self.dim_state)
        action = state_action[:,self.dim_state:].view(-1,self.dim_action)
        next_state = self.forward_model(state,action) # batch_size x dim_state
        next_action = self.policy(next_state)
        next_state_action = torch.cat((next_state,next_action),dim=-1)
        target_a_values = self.reward_normalized(state,action) + self.gamma*self.get_value_a(self.q_model,next_state_action) # batch_size x 1
        return target_a_values

    def compute_v_sa_model(self):
        v_sa = self.cross_approximate(fcn=self.get_v_sa, eps=self.eps_round_a, 
                         max_batch=self.max_batch_a, 
                        domain=self.domain_state_action, nswp=self.nswp_a, 
                        rmax=self.rmax_a, 
                        kickrank=self.kickrank_a, verbose=self.verbose)
        v_sa.round_tt(self.eps_cross_a)
        return v_sa.to(self.device)        


    def compute_dv_sa_model(self):
        dv_sa = self.cross_approximate(fcn=self.get_dv_sa, eps=self.eps_round_a, 
                         max_batch=self.max_batch_a, 
                        domain=self.domain_state_action, nswp=self.nswp_a, 
                        rmax=self.rmax_a, 
                        kickrank=self.kickrank_a, verbose=self.verbose)
        dv_sa.round_tt(self.eps_cross_a)
        return dv_sa.to(self.device)    

    def get_dv_sa(self,state_action):
        '''
            Given state_action=(s,a), get the output of value-function at the next state
            input: state-action pair, batch_size x (dim_state+dim_action)
        '''
        state = state_action[:,:self.dim_state].view(-1,self.dim_state)
        action = state_action[:,self.dim_state:].view(-1,self.dim_action)
        next_state = self.forward_model(state,action) # batch_size x dim_state
        dv_fcn = self.get_value_v(self.v_model,next_state)-self.get_value_v(self.v_model,state)+1e-9 # batch_size x 1
        return dv_fcn/self.dt

    def get_v_sa(self,state_action):
        '''
            Given state_action=(s,a), get the output of value-function at the next state
            input: state-action pair, batch_size x (dim_state+dim_action)
        '''
        state = state_action[:,:self.dim_state].view(-1,self.dim_state)
        action = state_action[:,self.dim_state:].view(-1,self.dim_action)
        next_state = self.forward_model(state,action) # batch_size x dim_state
        v_fcn = self.get_value_v(self.v_model,next_state)+1e-9 # batch_size x 1
        return v_fcn

    def get_advantage_from_value(self,state_action):
        '''
            Given state_action=(s,a), get the output of q-function using the current approximation of value fcn
            input: state-action pair, batch_size x (dim_state+dim_action)
        '''
        state = state_action[:,:self.dim_state].view(-1,self.dim_state)
        action = state_action[:,self.dim_state:].view(-1,self.dim_action)
        advantage = self.reward_normalized(state,action) + self.gamma*self.get_dv_sa(state_action) #(q_fcn-Vs)/self.dt# divide by dt?
        return advantage

    def get_advantage_from_a(self,state_action):
        '''
            Given state_action=(s,a), get the output of a-function using q
            input: state-action pair, batch_size x (dim_state+dim_action)
            R(s,a) + gamma*max_a' Q(s',a') - max_a Q(s,a)
        '''
        state = state_action[:,:self.dim_state].view(-1,self.dim_state)
        action = state_action[:,self.dim_state:].view(-1,self.dim_action)
        state_action = self.action_tt(tt_cores=self.policy_model_cores, 
                                                    domain=self.domain_state_action,
                                                    state=state)[:,0,:]
        next_state = self.forward_model(state,action)
        next_state_action = self.action_tt(tt_cores=self.policy_model_cores, 
                                                    domain=self.domain_state_action,
                                                    state=next_state)[:,0,:]
        q_fcn = self.get_value_a(self.q_model,state_action) # batch_size x 1
        q_fcn_next = self.get_value_a(self.q_model,next_state_action) # batch_size x 1
        
        advantage = self.reward_normalized(state,action) + self.gamma*q_fcn_next - q_fcn
        return advantage

    def policy_ttgo(self,state):
        '''
            Input: state, batch_size x dim_state
            Ouptut: best action, batch_size x dim_action
            Based on the current Advantage/Q-function
            Use TTGO to find armax_a advantage(s,a)
        '''
        # batch_size x n_samples x (dim_state+dim_action)
        state_action = self.action_tt(tt_cores=self.policy_model_cores, 
                                                    domain=self.domain_state_action,state=state)
        best_action = state_action[:,0,self.dim_state:] # batch_size x dim_action
        return best_action # batch_size x dim_action
    
    def policy_random(self,state):
        state_action = self.sample_action_random(state)
        q_values = self.get_value_a(self.policy_model,state_action.view(-1,self.dim_state_action)).view(state_action.shape[0],-1)
        # q_values =self.get_a_from_value(state_action.view(-1,self.dim_state_action)).view(state_action.shape[0],-1) # batch_size x n_samples 
        idx = torch.argmax(q_values,dim=-1).view(-1) # batch_size 
        best_action = state_action[torch.arange(state_action.shape[0]).to(self.device), idx][:,self.dim_state:] 
        return best_action # batch_size x dim_action

    def get_value_v(self, tt_model, state):
        y = get_value(tt_model=tt_model, x=state,  
                                domain=self.domain_state,
                                n_discretization=self.d_state,
                                device=self.device)
        return y


    def get_value_a(self, tt_model, state_action):
        y = get_value(tt_model=tt_model, x=state_action,  
                                domain=self.domain_state_action,
                                n_discretization=self.d_state_action, 
                                device=self.device)
        return y

    def get_value(self, tt_model, state, domain, n_discretization):
        ''' Evaluate the tt-model at the given state with Linear interpolation between the nodes'''
        y =  get_value(tt_model=tt_model, x=state,  
                                domain=domain,
                                n_discretization=n_discretization, 
                                max_batch=self.max_batch_a,
                                device=self.device)
        return y

    def get_tt_bounds(self,tt_model, domain):
        lb, ub = get_tt_bounds(tt_model, domain, device=self.device)
        return (lb,ub)

    def get_tt_bounds_v(self,tt_model):
        lb, ub = self.get_tt_bounds(tt_model, domain=self.domain_state)
        return (lb,ub)
    
    def get_tt_bounds_a(self,tt_model):
        lb, ub = self.get_tt_bounds(tt_model, domain=self.domain_state_action)
        return (lb,ub)

    def get_tt_mean(self, tt_model):
        '''
            find the mean of the tt-model
        '''
        tt_mean = get_tt_mean(tt_model)
        return tt_mean

    def canonicalize(self,tt_model):
        ''' Canonicalize the tt-cores '''
        tt_model.orthogonalize(0)
        return tt_model  

    
    def get_max_v(self, tt_model):
        return self.get_tt_max(tt_model=tt_model, domain=self.domain_state)

    def get_max_a(self, tt_model):
        return self.get_tt_max(tt_model=tt_model, domain=self.domain_state_action)


    def get_tt_max(self, tt_model, domain, n_samples=100):
        '''
        find the pseudo-max and argmax of a tt-model (absolute max) 
        '''
        max_, argmax_ = get_tt_max(tt_model, domain=domain,
         n_samples=n_samples, deterministic=True, device=self.device)
        return (max_, argmax_)
  
    
    def action_tt(self, tt_cores, domain, state):
        '''
        Consider the states to be continuous (linear interpolation between tt-nodes)
        state: batch_size x dim_state
        Generate n_samples (possible solutions/actions): argmax_a A(s,a); A: transformed advantage function (i.e A(s,a)>0)
        '''
        samples = deterministic_top_k(tt_cores=tt_cores, domain=domain, 
                                    x=state, 
                                    n_samples=self.n_samples, 
                                    n_discretization_x=self.d_state,
                                    device=self.device)
        return samples

    def sample_action_random(self, state):
        ''' sample from the uniform distribution from the domain '''
        samples_idx = torch.zeros([state.shape[0], 
                                self.n_samples, self.dim_state_action]).long().to(self.device)
        for site in range(self.dim_state_action):
            idxs = torch.multinomial(input=torch.tensor([1.]*self.d_state_action[site]).to(self.device), 
                                            num_samples=self.n_samples*state.shape[0], replacement=True)
            samples_idx[:,:,site] = idxs.reshape(state.shape[0],-1)
        idx_state = self.domain2idx(state)
        samples_idx[:,:,:self.dim_state] = idx_state[:,None,:].expand(-1,self.n_samples,-1)
        samples = self.idx2domain(samples_idx.view(-1,self.dim_state_action),
                                    self.domain_state_action).view(state.shape[0],
                                    self.n_samples,self.dim_state_action)
        samples[:,:,:state.shape[-1]] = state[:,None,:].expand(-1,self.n_samples,-1)
        return samples

            
    def clone(self):
        return copy.deepcopy(self)
    

    def fcn_batch_limited(self, fcn, max_batch):
        ''' To avoid memorry issues with large batch processing, reduce computation into smaller batches ''' 
        fcn_batch_truncated =  fcn_batch_limited(fcn=fcn, max_batch=max_batch, device=self.device)  
        return fcn_batch_truncated


    def cross_approximate(self, fcn,  max_batch, domain, rmax=200, nswp=10, 
                            eps=1e-3, verbose=False, kickrank=3):
        ''' 
            TT-Cross Approximation
            eps: precentage change in the norm of tt per iteration of tt-cross
         '''
        tt_model = cross_approximate(fcn=fcn,max_batch=max_batch, domain=domain, 
                        rmax=rmax, nswp=nswp, eps=eps, verbose=verbose, 
                        kickrank=kickrank, device=self.device)
        return tt_model.to(self.device)


    def get_elements(self, tt_model, idx):
        return get_elements(tt_model, idx)


    def idx2domain(self, I, domain): # for any discretization
        ''' Map the index of the tensor/discretization to the domain'''
        x = idx2domain(I, domain, device=self.device)
        return x

    def domain2idx(self, x):# non-uniform discretization
        ''' Map the states from the domain (a tuple of the segment) to the index of the discretization '''
        Idx = domain2idx(x, domain=self.domain_state, uniform=False, device=self.device) 
        return Idx


    def plt_training_stat(self):
        from matplotlib import pyplot as plt
        n_items = len(self.train_data.items())
        n_col = 3
        n_row = int(n_items/n_col+1)
        fig, axs = plt.subplots(n_row, n_col)
        fig.set_figheight(20)
        fig.set_figwidth(20)
        count = 0
        for key,value in self.train_data.items():
            axs[int(count/n_col), count%n_col].plot(value)
            axs[int(count/n_col), count%n_col].set_title(key)
            axs[int(count/n_col), count%n_col].grid()
            count+=1
            
        plt.show()

    def log_data(self):
        self.v_min, self.v_max = self.get_tt_bounds_v(self.v_model.clone())
        self.a_min, self.a_max = self.get_tt_bounds_a(self.a_model.clone())
        self.v_mean = self.get_tt_mean(self.v_model.clone())
        self.a_mean = self.get_tt_mean(self.a_model.clone())
        self.p_min, self.p_max = self.get_tt_bounds_a(self.policy_model.clone())
        self.p_mean = self.get_tt_mean(self.policy_model.clone())            

        self.train_data['v_norm'].append(self.v_model.norm().item())
        self.train_data['a_norm'].append(self.a_model.norm().item())
        self.train_data['p_norm'].append(self.policy_model.norm().item())


        self.train_data['v_mean'].append(self.v_mean)
        self.train_data['a_mean'].append(self.a_mean)
        self.train_data['p_mean'].append(self.p_mean)

        self.train_data['v_rank'].append((self.v_model.ranks_tt).max().item())
        self.train_data['a_rank'].append((self.a_model.ranks_tt).max().item())
        self.train_data['p_rank'].append((self.policy_model.ranks_tt).max().item())
        print("v_min: {:.2f}, v_mean: {:.2f}, v_max: {:.2f}".format(self.v_min, 
                                                    self.v_mean, self.v_max))
        print("a_min: {:.2f}, a_mean: {:.2f}, a_max: {:.2f}".format(self.a_min, 
                                                self.a_mean, self.a_max))
        print("p_min: {:.2f}, p_mean: {:.2f}, p_max: {:.2f}".format(self.p_min, 
                                                self.p_mean, self.p_max))
        print("Rank of V-model: ", (self.v_model.ranks_tt))
        print("Rank of A-model: ", (self.a_model.ranks_tt))
        print("Rank of P-model: ", (self.policy_model.ranks_tt))
    
    def save_model(self, file_name, i):
        torch.save({
        'policy_model':self.policy_model,
        'v_model':self.v_model,
        'a_model':self.a_model,
        'iter':i, 
        'reward_tt':self.reward_tt,
        'train_data':self.train_data
        },file_name+'.pt') # save the value model




