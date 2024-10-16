import torch

class PointMass:
    def __init__(self, position_min, position_max, velocity_min, velocity_max, action_min, action_max,
                    x_obst=[], r_obst=[], order=1, dim=2,dt=0.01,
                    w_obst=0., w_action=0.1, w_goal=0.9, w_scale=1.0, device='cpu'):
        ''' 
            dim: dimension of the space 
            state : (position) for velocity control  or (position, velocity) for acceleration control
            actions : position or velocity
        '''
        self.device = device
        self.dim=dim
        self.dim_state = dim
        self.dim_action = len(action_max)

        self.position_min = position_min.reshape(-1) 
        self.position_max = position_max.reshape(-1) 
        
        self.velocity_max = velocity_max.reshape(-1)
        self.velocity_min = velocity_min.reshape(-1)

        self.action_max = action_max
        self.action_min = action_min

        self.order=order

        if order==1: # velocity control
            self.fwd_simulate = self.fwd_simulate_1
        else: # acceleration control
            self.fwd_simulate = self.fwd_simulate_2

        self.x_obst = x_obst # positions of the spherical obstacles
        self.r_obst = r_obst # radii of the obstacles

        self.margin = 0.01
        self.dt = dt

        self.b_action = 0.1#2*(torch.linalg.norm(action_max)) #0.25*(torch.linalg.norm(action_max)**2)
        self.b_obst = 0.1#0.5
        self.b_goal = 0.1#2*(torch.linalg.norm(position_max)) #0.25*(torch.linalg.norm(position_max)**2)
        self.b_velocity = 1*(torch.linalg.norm(velocity_max))
        w_total = w_obst + w_goal +w_action
        self.w_obst = w_obst/w_total
        self.w_goal = w_goal/w_total # importance on state 
        self.w_action = w_action/w_total # importance of low control inputs
        self.w_scale = w_scale

        self.alpha_goal = self.w_goal/self.b_goal
        self.alpha_action = self.w_action/self.b_action
        self.alpha_obst = self.w_obst/self.b_obst

        self.target_position = torch.tensor([[0]*self.dim]).to(self.device)
        self.target_velocity = torch.tensor([[0]*self.dim]).to(self.device)

    def fwd_simulate_2(self, state, action):
        '''
        Given (state,action)  find the next state 
        state: (position,velocity)
        param: (mass, dt)
        action: acceleration
        '''
        position = state[:,:self.dim]
        velocity = state[:,self.dim:-1]
        mass = state[:,-1].view(-1,1)
        action = torch.clip(action, self.action_min, self.action_max)
        # dt = param[:,0].view(-1,1)
        
        
        d_position = velocity*self.dt
        d_velocity = mass*action*self.dt
        
       
        position_new = torch.clip(position+d_position, self.position_min, self.position_max)
        velocity_new = torch.clip(velocity+d_velocity, self.velocity_min, self.velocity_max)
        # collision_free = self.is_collision_free(position).view(-1,1)

        state_new = torch.cat((position_new,velocity_new,mass),dim=-1)    
        return state_new

    def fwd_simulate_1(self, state, action):
        '''
        Given (state,param, action) find the next state 
        state: position
        param: dt
        action: velocity
        '''
        dt = state[:,-1].view(-1,1)
        position = state[:,:-1]
        action = torch.clip(action, self.action_min, self.action_max)
        d_position = action*dt
        posiiton_new = torch.clip(position+d_position, self.position_min, self.position_max)
        state_new = torch.cat((posiiton_new,dt),dim=-1)
        return state_new


    def dist_position(self, position):
        # cost w.r.t. error to goal: (ex,ey) 
        ''' input shape: (batch_size, dim_state) '''
        d_x = torch.linalg.norm(position-self.target_position, dim=1)# batch_size x 1
        return d_x

    def dist_velocity(self, velocity):
        d_v = torch.linalg.norm(velocity-self.target_velocity, dim=1)# batch_size x 1
        return d_v

    def dist_collision(self,position):
        ''' 
            signed distance function (-1,inf), dist<0 inside the obstacle
            input shape: (batch_size,dim), find the distance into obstacle (-1 to inf), -1 completely inside, 0: at the boundary
        '''
        batch_size = position.shape[0]
        dist_obst = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
             dist_i = -1.0 + (torch.linalg.norm(position-self.x_obst[i], dim=1)/(self.r_obst[i]+self.margin)).view(-1)
             dist_obst = dist_obst + dist_i 
        return dist_obst.view(-1)

    def is_collision_free(self,position):
        ''' 
            input shape: (batch_size,dim), check if the state is in obstacle
        '''
        batch_size = position.shape[0]
        hit_obst = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
             dist_i = (torch.linalg.norm(position-self.x_obst[i], dim=1)/(self.r_obst[i]+self.margin)).view(-1)
             hit_obst = hit_obst + 1.0*(dist_i<1.)
        return (1.-hit_obst).view(-1)
    
    def dist_action(self,actions):
        '''
            Define the control cost
            input_shape = (batch_size, dim_action)
        '''
        d_action = torch.linalg.norm(actions,dim=-1)**2
        return d_action        


    def reward_state_action(self, state,action):
        ''' 
            Compute the stage reward given the action (ctrl input) and the state
        '''
        state=state[:,:-1]
        dim_state = state.shape[1]
        if self.order == 1:
            position = state.view(state.shape[0],dim_state)
            action = action.view(action.shape[0],self.dim_action)
            d_goal = (self.dist_position(position)).view(-1)
            r_goal = -1*(d_goal/self.b_goal)#2*(-0.5 + 1/(1+d_goal/self.b_goal))#-1 + torch.exp(-(d_goal/self.b_goal))# (0,1) #  -torch.log(1e-2+d_goal)#   2*(-0.5+1/(1+d_goal)) #    


        else:
            position = state[:, :self.dim]
            velocity = state[:,self.dim:]
            action = action.view(action.shape[0],self.dim) #acceleration
            d_goal = (self.dist_position(position)).view(-1)
            d_velocity = (self.dist_position(velocity)).view(-1)
            r_pos =  -1*(d_goal/self.b_goal)#2*(-0.5 + 1/(1+d_goal/self.b_goal))#-1 + torch.exp(-(d_goal/self.b_goal))# (0,1) #  -torch.log(1e-2+d_goal)#   2*(-0.5+1/(1+d_goal)) #    
            r_vel = -1*(d_velocity/self.b_velocity)
            r_goal = 1*r_pos + 0.00*r_vel

        d_obst = (self.dist_collision(position)).view(-1) # range:(-1,inf), where (-1,0): within the obst, (0,inf): away from obstacle
        r_obst = (((d_obst/self.b_obst))*(d_obst<0))#-1 + torch.sigmoid(d_obst/0.01) # (-1,0)
        
        d_action = (self.dist_action(action)).view(-1)
        r_action = -1*(d_action/self.b_action) #-1 + torch.exp(-d_action/self.b_action) #-1+torch.exp(-(d_action/self.b_action)) # -1 + torch.exp(-d_action**2) #torch.exp(-d_action**2) # (-1,0) #-torch.log(1e-2+d_action)#

        r_total = self.w_scale*(r_goal*self.w_goal+ r_obst*self.w_obst + r_action*self.w_action)
        r_all = torch.cat(( r_goal.view(-1,1), r_action.view(-1,1), r_obst.view(-1,1)),dim=1)
        return r_total, r_all
 

################################################################################################################
################################################################################################################
class MiniGlof:
    '''
        Dynamics of min-golf
        state: (theta, dtheta) # (joinr_angle,joint_vel), joint_angle in (-pi, pi), it is 0 at the stable equilibrium
        action: (u) # joint torquw
    '''
    def __init__(self, state_min, state_max, action_max, action_min, dt=0.01, w_scale=1.0, w_goal=0.9,w_action=0.1,device='cpu'):

        self.g = 9.81
        self.dt=dt

        self.state_min = state_min
        self.state_max = state_max # max of ()
        self.action_max = action_max
        self.action_min = action_min

        w_total = w_goal+w_action
        self.w_goal = w_goal/w_total
        self.w_action = w_action/w_total

        self.w_scale = w_scale
        self.gamma = 0 #elasticity coefficient


    def fwd_simulate(self, state_param,action):
        '''
        state: (x, y, vx, vy)
        param: (mass, mu)
        action: (a_x, a_y)
        '''
        state = state_param[:,:4]
        param = state_param[:,4:]
        mass = param[:,0].view(-1,1)
        mu = param[:,1].view(-1,1)
        position = state[:,:2]
        velocity = state[:,2:]
        action = torch.clip(action, self.action_min, self.action_max)

        
        acc = -mu*9.81*velocity/torch.linalg.norm(velocity,dim=-1).view(-1,1)
        new_position = position + velocity*self.dt
        # new_velocity = action/mass + velocity + acc*self.dt
        new_velocity = action/mass + velocity + acc*self.dt
        state_new = torch.cat((new_position, new_velocity),dim=-1)
        state_new = torch.clip(state_new, self.state_min, self.state_max)
        return torch.cat((state_new, param),dim=-1)

    def reward_state_action(self,state_param,action):
        state = state_param[:,:4]
        position = state[:,:2]

        r_goal = -1*(torch.linalg.norm(position,dim=-1)/0.01)**2 # -self.state_max[0]*10
        r_action = -1* torch.linalg.norm(action,dim=-1)**2
        return r_goal*100 #+ r_action*0.01
        # return r_goal


################################################################################################################
################################################################################################################
class Reorientation:
    '''
        Dynamics of a single pendulum
        state: (theta, dtheta) # (joinr_angle,joint_vel), joint_angle in (-pi, pi), it is 0 at the stable equilibrium
        action: (u) # joint torquw
    '''
    def __init__(self, state_min, state_max, action_max, action_min, dt=0.01, w_scale=1.0, w_goal=0.9,w_action=0.1,device='cpu'):

        self.g = 9.81
        self.dt=dt

        self.state_min = state_min
        self.state_max = state_max # max of ()
        self.action_max = action_max
        self.action_min = action_min

        w_total = w_goal+w_action
        self.w_goal = w_goal/w_total
        self.w_action = w_action/w_total
        self.dim_state= len(state_max)
        self.dim_action = len(action_max)

        self.w_scale = w_scale
        self.device = device
        self.gamma = 0 #elasticity coefficient


    def forward_simulate(self, state_param, action):
        '''
        state: (theta,dtheta)
        param: (mass, length, mu_torsion)
        action: torque
        '''
        # action = torch.clip(action,self.action_min,self.action_min)
        theta = state_param[:,0].view(-1,1)
        ctheta = torch.cos(theta)
        stheta = torch.sin(theta)
        dtheta = state_param[:,1].view(-1,1)

        param = state_param[:,2:].view(-1,3) # mass, length, mu

        
        mgl = (param[:,0]*param[:,1]*9.81).view(-1,1) # m*l*g

        I = 1/3*param[:,0]*(2*param[:,1])**2

        tau_gravity  = mgl*stheta#*torch.sign(theta[:, 0])
        tau_friction = 2* param[:,2]*(action[:,0]**(1+self.gamma))#*torch.sign(dtheta[:, 0]) # torsional friction

        ddtheta = (-tau_gravity - tau_friction[:, None])/I[:, None]

        theta = theta + dtheta*self.dt

        dtheta = dtheta + ddtheta*self.dt

        state_new = torch.concat((theta.view(-1,1),dtheta.view(-1,1)),dim=-1)

        state_new = torch.clip(state_new, self.state_min[:2], self.state_max[:2])
        return torch.cat((state_new, param),dim=-1)

    def reward_state_action(self,state_param,action):
        state = state_param[:,:2]
        height = -1*torch.cos(state[:,0])
        
        r_goal = -1*((state[:,0].abs()-torch.pi).abs())/0.01 # -self.state_max[0]*10
        r_action = -1* torch.linalg.norm(action,dim=-1)

        return r_goal*100 + r_action