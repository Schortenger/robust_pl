import torch
import tntorch as tnt
import numpy as np
import scipy
torch.set_default_dtype(torch.float64)

import math 

def get_exponential_discretization(xmax=1.0,n=100,sc=1.0,flip=False,device='cpu'):
    '''
    note: applies only for symmetric bounds; (-xmax,xmax)
    Generates discretization non-uniformly: in an exponential manner
    
    sc --> inf => uniform discretization
    sc --> 0 => discretization points are more dense near 0.
    flip=True:
        discretization points are more dense near the boundary
    '''

    xmin = -1*xmax
    even_n = n%2
    n_p = int(n/2) + even_n
    xp = xmax*torch.linspace(0,1,n_p).to(device)
    if not flip:
        yp = -1+torch.exp(sc*xp.abs()/xmax)
    else:
        yp = 1-1/(1+torch.exp(sc*xp.abs()/xmax))
        yp = yp-yp.min()
    yp = xmax*yp/yp.max()
    idx_n = -1*torch.arange(n_p)[even_n:].to(device)
    xn = -1*xp[idx_n]
    yn = -1*yp[idx_n]
    y = torch.cat((yn,yp),dim=-1)
    return y

def idx2domain(I, domain, device): # for any discretization
    ''' Map the index of the tensor/discretization to the domain'''
    X = torch.zeros(I.shape).to(device)
    for i in range(I.shape[1]):
        X[:,i] =  domain[i][I[:, i]]
    return X

def domain2idx(x, domain, device, uniform=False):
    ''' 
    Map x from the domain to the index of the discretization 
    '''
    I = torch.zeros(x.shape).to(device)

    if uniform: # if the discretization is uniform
        for i in range(x.shape[-1]):
            min_i = domain[i][0] # 
            step_i = domain[i][1]-domain[i][0] 
            I[:,i] = ((x[:,i] - min_i)/step_i).round() 
    else: 
        for i in range(x.shape[-1]):
            I[:,i] = torch.argmin(torch.abs(x[:,i].view(-1,1)- domain[i]), dim=1) 
    return I.long()

def get_elements_from_cores(tt_cores, idx):
    '''
    Given the tt_cores and a batch of index get the  elements
    '''
    mat_ = tt_cores[0][:,idx[:,0],:]
    for i in range(1,idx.shape[-1]):
        mat_ = torch.einsum('ijk,kjl->ijl',(mat_,tt_cores[i][:,idx[:,i]]))
    return mat_.view(-1)


def get_elements(tt_model, idx):
    '''
    Given the tt_model in tntorch format and a batch of index get the  elements
    '''
    return get_elements_from_cores(tt_model.tt().cores, idx)


def get_tt_mean(tt_model):
    '''
    Given the tt_model in tntorch format find the mean of the tt-model
    '''

    return get_tt_mean_from_cores(tt_model.tt().cores)


def get_tt_mean_from_cores(tt_cores):
    '''
        find the mean of the tt-model given its cores
    '''
    sum_ = tt_cores[0].sum(dim=1)/tt_cores[0].shape[1]
    for core in tt_cores[1:]:
        sum_ = sum_@core.sum(dim=1)/core.shape[1]
    return sum_.item()



def get_value(tt_model, x,  domain, 
                    n_discretization, max_batch=10**5, device="cpu"):
    ''' 
    Evaluate the tt-model (in tntorch format) at the given state with Linear interpolation between the nodes. Assumes uniform discretization 
    dh_domain : a 1D tensor containing the step size of discretization for each site/mode
    n_discretization: a 1D tensor continginingthe number of discretization points along each mode
    '''
    return get_value_from_cores(tt_model.tt().cores, x,  domain, 
                                    n_discretization,
                                    max_batch, device)

def get_value_from_cores(tt_cores, x,  domain, 
                                    n_discretization=None, max_batch=10**5,
                                     device="cpu"):
    
    if n_discretization is None:
        n_discretization = torch.tensor([len(dom) for dom in domain]).to(device)
    
    def fcn(x_batch):
        idx_1 = domain2idx(x_batch, domain=domain, device=device) # find the closest/floor index of the state (w.r.t to the discretizaton)
        x_1 = idx2domain(idx_1, domain, device=device) 
        dx = (x_batch-x_1)#/dh_domain.view(1,-1) # 
        idx_2 = torch.clip(idx_1+torch.sign(dx),
                                    n_discretization[:x_batch.shape[-1]]*0,
                                    n_discretization[:x_batch.shape[-1]]-1).long() # next index
        x_2 = idx2domain(idx_2, domain, device=device)
        dx = dx.abs()/(1e-6+(x_2-x_1).abs())
        mat_ = tt_cores[0][:,idx_1[:,0],:]+dx[:,0].view(1,-1,1)*(tt_cores[0][:,idx_2[:,0],:]-tt_cores[0][:,idx_1[:,0],:])
        for i in range(1,idx_1.shape[-1]):
            mat = tt_cores[i][:,idx_1[:,i],:]+dx[:,i].view(1,-1,1)*(tt_cores[i][:,idx_2[:,i],:]-tt_cores[i][:,idx_1[:,i],:])
            mat_ = torch.einsum('ijk,kjl->ijl',mat_,mat)
        return mat_.view(-1)
    return fcn_batch_limited(fcn=fcn,max_batch=max_batch, device=device)(x)

def get_value_from_cores_nonbatch(tt_cores, x,  domain, 
                                    n_discretization=None, device="cpu"):
    ''' 
    Evaluate the tt-model (given its tt_cores) at the given state with 
    Linear interpolation between the nodes.  
    dh_domain : a 1D tensor containing the step size of discretization for each site/mode
    n_discretization: a 1D tensor continginingthe number of discretization points along each mode
    '''
    if n_discretization is None:
        n_discretization = torch.tensor([len(dom) for dom in domain]).to(device)

    idx_1 = domain2idx(x, domain=domain, device=device) # find the closest/floor index of the state (w.r.t to the discretizaton)
    x_1 = idx2domain(idx_1, domain, device=device) 
    dx = (x-x_1)#/dh_domain.view(1,-1) # 
    idx_2 = torch.clip(idx_1+torch.sign(dx),
                                n_discretization[:x.shape[-1]]*0,
                                n_discretization[:x.shape[-1]]-1).long() # next index
    x_2 = idx2domain(idx_2, domain, device=device)
    dx = dx.abs()*1/(1e-6+(x_2-x_1).abs())
    mat_ = tt_cores[0][:,idx_1[:,0],:]+dx[:,0].view(1,-1,1)*(tt_cores[0][:,idx_2[:,0],:]-tt_cores[0][:,idx_1[:,0],:])
    for i in range(1,idx_1.shape[-1]):
        mat = tt_cores[i][:,idx_1[:,i],:]+dx[:,i].view(1,-1,1)*(tt_cores[i][:,idx_2[:,i],:]-tt_cores[i][:,idx_1[:,i],:])
        mat_ = torch.einsum('ijk,kjl->ijl',mat_,mat)
    return mat_.view(-1)

def get_value_discrete(tt_model, x, domain, device="cpu"):
    '''
        Evaluate tt-model at the given point (in batch) from the domain. Assuming uniform discretization
        Input: x, batch_size x dim
    '''
    idx_state = domain2idx(x, domain, device) # find the index (w.r.t to the discretizaton)
    return get_elements(tt_model,idx_state).view(-1) #v_model[idx_state].torch() # batch_size x 1


def cross_approximate(fcn,  max_batch, domain, 
                        rmax=200, nswp=20, eps=1e-4, verbose=False, 
                        kickrank=3, return_info=False, device="cpu"):
    ''' 
        TT-Cross Approximation using tntorch's implementation
        eps: accuracy of approximation
    '''
    if return_info:
        tt_model, info = tnt.cross(fcn_batch_limited(fcn, max_batch=max_batch, device=device),
            domain=domain,
            max_iter=nswp, eps=eps, rmax=rmax, kickrank=kickrank, 
            function_arg='matrix',device=device,_minimize=False,
            val_size=1e5, verbose=verbose, return_info=return_info)
        tt_model.round_tt(eps)
        return (tt_model.to(device), info)
    else:

        tt_model = tnt.cross(fcn_batch_limited(fcn, max_batch=max_batch, device=device),
            domain=domain,
            max_iter=nswp, eps=eps, rmax=rmax, kickrank=kickrank, 
            function_arg='matrix',device=device,_minimize=False,
            val_size=1e5, verbose=verbose, return_info=return_info)
        tt_model.round_tt(eps)
        return tt_model.to(device)


def fcn_batch_limited(fcn, max_batch=10**5, device="cpu"):
    ''' 
    To avoid memorry issues with large batch processing, 
    reduce computation into smaller batches 
    '''   
    def fcn_batch_truncated(x):
        batch_size = x.shape[0]
        fcn_values = torch.empty(batch_size).to(device)
        num_batch = batch_size//max_batch
        end_idx = 0
        for i in range(num_batch):
            start_idx = i*max_batch
            end_idx = (i+1)*max_batch
            fcn_values[start_idx:end_idx] = fcn(x[start_idx:end_idx].view(-1,x.shape[1]))
        if batch_size>end_idx:          
            fcn_values[end_idx:batch_size] = fcn(x[end_idx:batch_size].view(-1,x.shape[1]))
        return fcn_values
    return fcn_batch_truncated


def sample_random(batch_size, n_samples, domain, device="cpu"):
    ''' sample from the uniform distribution from the domain '''
    samples = torch.empty((batch_size,n_samples)).to(device)
    for i in range(len(domain)):
        samples[:,i] = domain[i][0] + (domain[i][-1]-domain[i][0])*torch.rand(size=(batch_size,n_samples))
    return samples


def stochastic_choice(M, alpha=0.99, rand_state=None, device="cpu"):
    '''
        Given pmf get the prioritized samples
        M: batch_size x n_samples x n  
        Treat each row of a matrix M[:,i,:] as a PMF and select a column per row according to it
    '''
    
    #filtering low pmf samples
    if rand_state is not None:
        torch.random.manual_seed(torch.randn(1).data)
    M= torch.abs(M) # batch_size x n_samples x n_site
    M_max, _ = torch.max(M,dim=-1) # batch_size x n_samples
    M_min, _ = torch.min(M,dim=-1)
    M_mean = M.mean(dim=-1)
    
    M_threshold = M_mean + alpha*(M_max-M_mean)
    
    M_max  = M_max[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    M_min  = M_min[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    M_mean  = M_mean[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    M_threshold  = M_threshold[:,:,None].expand(-1,-1,M.shape[-1]) # batch_size x n_samples x n_site
    
    M = M*(M>M_threshold)        
    M= M/(1e-9+M_max) # batch_size x n_samples x n_site

    M=M**(1/(1e-9+1-alpha))  # higher density is given higher importance
    M=M+1e-9
    M = M/(torch.sum(M, dim=-1)[:,:, None]) + 1e-9  # Normalize the pdf, batch_size x n_samples x n_site
    samples = torch.multinomial(M.view(-1,M.shape[-1]),1).view(M.shape[0],-1) # (batch_size*n_samples) x 1        
    if rand_state is not None:
        torch.random.set_rng_state(rand_state)
    return samples # batch_size x n_samples


def deterministic_choice(M,n_samples,idx_site, device="cpu"):
    """
        M: batch_size x n_samples x n_site

    """
    idx_site[:,1:] = (idx_site[:,1:]-idx_site[:,:-1]).abs()>0 
    idx_site[:,0] = 1
    
    bs = M.shape[0]
    n_site = M.shape[2]
    M = M*idx_site[:,:,None].expand(-1,-1,n_site) # make pmf corresponding to repeated indices to be zeo
    next_site = torch.zeros(bs,n_samples).to(device)
    previous_sample_id = torch.zeros(bs,n_samples).to(device)
    M2d = M.view(bs,-1) # bs x (n_samples*n_site)
    idx_k = torch.topk(M2d, k=n_samples, dim=-1)[1] # bs x n_samples
    next_site[:,:n_samples] = (idx_k).fmod(n_site) # which site next, bs x n_samples
    previous_sample_id[:,:n_samples] = (idx_k/n_site).long() # previous sample_id, b_size x n_samples
    return next_site.long(), previous_sample_id.long()


def contract_sites(tt_model, site_x, pro_x, device, eps=1e-6):
    '''
    Contract the cores of the tt-model given the weights for each discretization point 
    corresponding to each of the contracted site.
    p_x: a list of 1D tensor (probaility of each index of the site) 
    Return a contracted model 
    '''
    p_x = pro_x.clone()
    tt_cores = [core for core in tt_model.tt().cores[:]] # r_k x n_k x r_kn
    mat =  (tt_cores[site_x[0]]*(p_x[0].view(1,-1,1))).sum(dim=1) # r_k x 1 x r_kn
    for i,site in enumerate(site_x[1:]):
        mat_i = (tt_cores[site]*(p_x[1+i].view(1,-1,1))).sum(dim=1)
        mat = mat@mat_i
    state_id = site_x[-1]+1
    tt_cores_c = tt_cores[:site_x[0]] + tt_cores[state_id:]
    if site_x[-1] < len(tt_cores_c):
        tt_cores_c[site_x[0]] = torch.einsum('ij,jkl->ikl',mat,tt_cores_c[site_x[0]])
    else:
        last_state_id = site_x[0]-1
        tt_cores_c[last_state_id] = torch.einsum('ikj,jl->ikl',tt_cores_c[last_state_id],mat)

    tt_c_model = tnt.Tensor(tt_cores_c)
    # tt_c_model.round_tt(eps=eps)
    return tt_c_model.to(device)


def get_prob_x(mean_id, site_x, n_param, sigma=0.1, length=1, flag = 'uniform', device='cpu'):
    """
    param: 
    mean_id: index of the true parameter within parameter domain
    site_x: the dimension of the site to be contracted
    n_param: number of discretization points in parameter domain, (num_param x n_param)

    Given a rough guess of the true parameter, we assume the true parameter respects a probability distribution around the guess
    sigma: covariance if gaussian distribution
    length: width range if uniform distribution
    flag: 'gaussian' or 'uniform' #default is uniform
    
    return: p_x: probability of each discretization point in the parameter domain

    """

    num_param = len(site_x) 
    p_x = torch.zeros(num_param, n_param).to(device)

    if flag == 'gaussian':
        for id in range(num_param):
            mu = mean_id[id].cpu().numpy()
            values = np.arange(0, n_param.cpu().numpy())

            # compute probabilities and normalize
            probabilities = scipy.stats.norm.pdf(values, mu, sigma)
            probabilities /= np.sum(probabilities)
            p_x[id, :] = torch.tensor(probabilities).to(device)

    elif flag == 'uniform':
        length = max(int(length), 1)
        for id in range(num_param):
            mu = mean_id[id].cpu().numpy()
            probabilities = np.zeros(n_param)
            probabilities[mu:mu+length] = 1
            probabilities /= np.sum(probabilities)

            p_x[id, :] = torch.tensor(probabilities).to(device)
    return p_x


def prob_sites(tt_model, site_x, p_x, device, eps=1e-6):
    '''
    Contract the cores of the tt-model given the weights for each discretization point 
    corresponding to each of the contracted site.
    p_x: a list of 1D tensor (probaility of each index of the site) 
    Return a contracted model 
    '''

    tt_cores = [core for core in tt_model.tt().cores[:]] # r_k x n_k x r_kn
    mat =  (tt_cores[site_x[0]]*(p_x[0].view(1,-1,1))).sum(dim=1) # r_k x 1 x r_kn
    for i,site in enumerate(site_x[1:]):
        mat_i = (tt_cores[site]*(p_x[1+i].view(1,-1,1))).sum(dim=1)
        mat = mat@mat_i
    state_id = site_x[-1]+1
    tt_cores_c = tt_cores[:site_x[0]] + tt_cores[state_id:]
    # if site_x[-1] < len(tt_cores):
    tt_cores_c[site_x[0]] = torch.einsum('ij,jkl->ikl',mat,tt_cores_c[site_x[0]])
    # else:
    #     tt_cores_c[-1] = torch.einsum('ikj,jl->ikl',tt_cores_c[-1],mat)
    tt_c_model = tnt.Tensor(tt_cores_c)
    # tt_c_model.round_tt(eps=eps)
    return tt_c_model.to(device)


def contract_site(tt_model, site_x, p_x, device, eps=1e-6):
    '''
    Contract the cores of the tt-model given the weights for each discretization point 
    corresponding to each of the contracted site.
    p_x: a list of 1D tensor (probaility of each index of the site) 
    Return a contracted model 
    '''

    tt_cores = [core for core in tt_model.tt().cores[:]] # r_k x n_k x r_kn
    mat =  (tt_cores[site_x[0]](p_x[1+i].view(1,-1,1))).sum(dim=1) 
    for i,site in enumerate(site_x[1:]):
        mat_i = (tt_cores[site]*(p_x[1+i].view(1,-1,1))).sum(dim=1)
        mat = mat@mat_i
    tt_cores_c = tt_cores[:site_x[0]] + tt_cores[site_x[-1]:]
    if site_x[-1] < len(tt_cores):
        tt_cores_c[site_x[-1]] = torch.einsum('ij,jkl->ikl',mat,tt_cores_c[site_x[-1]])
    else:
        tt_cores_c[-1] = torch.einsum('ikj,jl->ikl',tt_cores_c[-1],mat)
    return tnt.Tensor(tt_cores_c).round_tt(eps=eps).to(device)



def condition_site(tt_cores, x, domain_x, n_discretization_x,device):
    '''
    Condition (or slicing) the cores of the tt-model given the values corresponding to a site. 
    Assumes x: batch_size x dim_x correspond to the first few cores
    Return the conditioned model:  tt_cores of shape batch_size x r_i x n_i x r_i' 
    '''
    batch_size = x.shape[0]
    dim_x = x.shape[1]
    # interpolate to find the corresponding slice for x
    idx_x = domain2idx(x,domain_x,device).view(batch_size,-1) # batch_size x dim_state
    x_1 = idx2domain(idx_x,domain_x,device) 
    dx = (x - x_1)
    idx_x_next = torch.clip(idx_x+torch.sign(dx),n_discretization_x*0,n_discretization_x-1).long() # next index (w.r.t disctretization)
    x_2 = idx2domain(idx_x_next,domain_x,device) 
    dx = torch.abs(dx)*1.0/(1e-6+(x_2-x_1).abs())
    # interpolate between the adjacent slices 
    for site in range(x.shape[-1]):
        tt_cores[site] = (tt_cores[site][:,idx_x[:,site],:]+dx[:,site].view(1,-1,1)*(tt_cores[site][:,idx_x_next[:,site],:]-tt_cores[site][:,idx_x[:,site],:]))
    # tranform cores so that it is: batch_size x r_k x -1 x r_kn     
    tt_cores_ext = [tt_cores[site][None,:,:,:].permute(2,1,0,3) for site in range(dim_x)]+[tt_cores[site][None,:,:,:].expand(batch_size,-1,-1,-1) for site in range(dim_x,len(tt_cores))]
    # Merge the slices corresponding to x into one core of size: b_state x 1 x 1 x r and then merge it to the non-sliced core b_state x 1 x n_a x r_a
    core_state = tt_cores_ext[0]
    for site in range(1,dim_x):
        core_state = torch.einsum('bijk,bkjl->bijl',core_state,tt_cores_ext[site])
    tt_cores_ext[dim_x] = torch.einsum('bi,ijk->bjk',core_state[:,0,0,:],tt_cores[dim_x])[:,None,:,:] # b_state x 1 x n_1 x r
    tt_cores_ext = tt_cores_ext[dim_x:]
    return tt_cores_ext # each core is of shape barch_size x r_ x n_ x r and the number of cores is len(tt_cores)-dim_x

def get_rights(tt_cores_ext, device):
    batch_size = tt_cores_ext[0].shape[0]
    # batch_size x r_k x r_kn
    tt_cores_action_summed =[torch.sum(core,dim=2) for core in tt_cores_ext] # batch_size x r_k x r_kn 
    rights = [torch.ones(batch_size,1).to(device).view(-1,1)] # each element is batch_size x r_k
    for site, summed_core in enumerate(tt_cores_action_summed[::-1]):
        r_ = torch.einsum('ijk,ik->ij',summed_core, rights[-1])
        rights.append(r_) # batch_size x r_k : batch_size x (r_k x r_kn) times (batch_size x r_kn)
    rights = rights[::-1] # batch_size x r_k
    return rights


def stochastic_top_k(tt_cores, domain, 
                         n_discretization_x=None, x=None, n_samples = 1, 
                         alpha=0.9, device="cpu", train=True):
    '''
    Consider x to be continuous (linear interpolation between tt-nodes)
    state: batch_size x dim_state
    Generate n_samples points from Q-function (treated as a joint PDF distribution ) 
    '''
    dim = len(tt_cores)
    
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)

    rights = get_rights(tt_cores_ext,device=device)

    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #
    lefts = torch.ones([batch_size, n_samples, 1]).to(device) # batch_size x n_samples x 1
    for site in range(len(tt_cores_ext)):
        fiber = torch.einsum('ijkl,il->ijk', (tt_cores_ext[site], rights[site+1])) # batch_size x r_k x n_k
        pmf = torch.einsum('ijk,ikl->ijl', (lefts, fiber)) # batch_size x n_samples x n_site 
        samples_idx[:,:, site] = stochastic_choice(M=pmf, alpha=alpha, rand_state=None, device=device ) # batch_size x n_samples
        core_sliced = (tt_cores_ext[site].permute([0,2,1,3])[torch.arange(tt_cores_ext[site].shape[0]).unsqueeze(1),samples_idx[:,:, site]]).permute([0,2,1,3])
        lefts = torch.einsum('ijk,ikjl->ijl', (lefts, core_sliced))

        
    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(tt_cores_ext):], device).view(batch_size,n_samples,len(tt_cores_ext))
    if x is not None:
        samples_concat = torch.concat((x[:,None,:].expand(-1,n_samples,-1),samples),dim=-1)
    else:
        samples_concat = samples
    return samples_concat


def deterministic_top_k(tt_cores, domain=[], 
                x=None, n_samples=100, 
                n_discretization_x=None, 
                device="cpu", train=True):
    '''
    Consider the states to be continuous (linear interpolation between tt-nodes)
    x: batch_size x dim_x (task variables)
    Generate n_samples points from tt-model (treated as a joint PDF distribution ) corresponding to top-k max values
    The tt_cores are not assumed to be right orthogonalized (orthogonal model).
    If not, call canonlicalize(tt_model) prior to calling this method 
    This will speed up the process 
    '''
    dim = len(tt_cores)
    if x is None: # no task variable means no conditioning
        batch_size = 1
        tt_cores_ext = [core[None,:,:,:] for core in tt_cores]
    else:
        if n_discretization_x is None:
            n_discretization_x = torch.tensor([len(domain[i]) for i in range(x.shape[-1])]).to(device)
        batch_size = x.shape[0]
        tt_cores_ext = condition_site(tt_cores=tt_cores[:], x=x, 
                            domain_x=domain[:x.shape[-1]], 
                            n_discretization_x=n_discretization_x, 
                            device=device)


    # rights = get_rights(tt_cores_ext, device=device)
    samples_idx = torch.zeros([batch_size, n_samples, len(tt_cores_ext)]).long().to(device) #

    # pmf:  batch_size x 1 x n
    pmf = torch.linalg.norm(tt_cores_ext[0],dim=-1) # tt_cores_ext[0]: batch_size X 1 X n X r_1
    # pmf = torch.einsum('ijkr,ir->ijk',tt_cores_ext[0],rights[1]).abs()

    n_site_0 = tt_cores_ext[0].shape[-2]
    # samples_site:  batch_size x min(n_samples,n_site) 
    idx_k = torch.topk(pmf.view(batch_size,-1),k=min(n_samples,n_site_0),dim=-1)[1].fmod(n_site_0).long()
    if n_site_0 < n_samples: 
        samples_idx[:,:,0] = idx_k.repeat(1,int(n_samples/n_site_0)+1)[:,:n_samples] #batch_size x n_samples
    else:
        samples_idx[:,:,0] = idx_k
    # p_cum: batch_size x n_samples x r_1
    p_cum = (tt_cores_ext[0].permute([0,2,1,3])[torch.arange(batch_size).unsqueeze(1),idx_k]).permute([0,2,1,3])[:,0,:,:]

    for site in range(1,len(tt_cores_ext)):
        n_sites = tt_cores_ext[site].shape[-2]

        pmf_pre = torch.einsum('ijk,iklm->ijlm', (p_cum, tt_cores_ext[site])).flatten(1,2)#.view(batch_size,-1,tt_cores_ext[site].shape[-1]) # batch x n_site*n_samples x r_site
        pmf = torch.linalg.norm(pmf_pre,dim=-1) # batch x (n_site*n_samples) 
        # pmf = torch.einsum('ijr,ir->ij',pmf_pre,rights[site+1]).abs()
        idx_k = torch.topk(pmf, k=n_samples, dim=-1)[1].long() # bs x n_samples
        
        samples_idx[:,:,site] = idx_k.fmod(n_sites).long()#((idx_k)/n_samples).floor().long()#( # top-k indices from the site, bs x n_samples
        samples_prev_id  = (idx_k/n_sites).long()#(idx_k).fmod(n_samples).long()#idx_k - samples_idx[:,:,site]*n_sites # ((idx_k-1)/n_sites).long() # update previous site index
        samples_idx[:,:,:site]= samples_idx[:,:,:site][torch.arange(batch_size).unsqueeze(1),samples_prev_id]
        # p_cum: batch_size x  n_samples  x r_site 
        p_cum = pmf_pre[torch.arange(batch_size).unsqueeze(1),idx_k]

    
    samples = idx2domain(samples_idx.flatten(0,1),domain[-len(tt_cores_ext):], device).view(batch_size,n_samples,len(tt_cores_ext))
    if x is not None:
        samples_concat = torch.concat((x[:,None,:].expand(-1,n_samples,-1),samples),dim=-1)
    else:
        samples_concat = samples

    return samples_concat


def get_tt_max(tt_model, domain, n_samples=100, deterministic=True, alpha=0.9, device="cpu"):
    '''
    Note: max is w.r.t the absolute value
    find the pseudo-max and argmax of a tt-model (absolute max) in a stochastic way
    '''
    tt_model_o =  tt_canonicalize(tt_model)
    tt_cores = tt_model_o.tt().cores[:]
    # Warm-up for mass sampling
    if deterministic:
        samples = deterministic_top_k(tt_cores=tt_cores, 
                        n_samples=n_samples, 
                        domain=domain, 
                        device=device)
    else:
        samples = stochastic_top_k(tt_cores=tt_cores, 
                        n_samples=n_samples, alpha=alpha,
                        domain=domain, device=device)
    samples_idx = domain2idx(samples.flatten(0,1),domain,device)
    values = get_elements(tt_model_o,samples_idx)
    idx = torch.argmax(torch.abs(values)) # batch_size 
    best_value = values[idx]

    return best_value, samples_idx[idx].view(-1) # max, argmax



def get_tt_bounds(tt_model,domain,device="cpu"):
    tt_model_1 = tt_model.clone()
    bound_1, idx_1 = get_tt_max(tt_model, domain, device=device)
    bound_1  =  get_elements(tt_model,idx_1.view(1,-1)).item()
    tt_model_2 = tt_model_1-bound_1
    tt_model_2.round_tt(eps=1e-9)
    bound_2, idx_2 = get_tt_max(tt_model_2.to(device),domain, device=device)
    bound_2  = get_elements(tt_model,idx_2.view(1,-1)).item()
    upper_bound = bound_1 if (bound_1>bound_2) else bound_2
    lower_bound = bound_1 if (bound_1<bound_2) else bound_2
    return (lower_bound,upper_bound)


def normalize_tt(tt_model, domain, lb=1., ub=100., 
                    auto_bound=True,canonicalize=True,
                    device="cpu"):
    lower_bound, upper_bound  = get_tt_bounds(tt_model, domain, device=device)
    if auto_bound:
        lb = 1 + upper_bound - lower_bound
        tt_model_out = lb + (tt_model.to("cpu")-lower_bound)
    else:
        tt_model_out = lb + (tt_model.to("cpu")-lower_bound)*((ub-lb)/(upper_bound-lower_bound))
    if canonicalize:
        tt_model_out = tt_canonicalize(tt_model_out)
    else:
        tt_model_out.round_tt(eps=1e-9) # not necessary
    return tt_model_out.to(device)

def tt_canonicalize(tt_model,site=0):
    ''' 
    Return an  orthogonalized tt-model at site. 
    For i>site, torch.einsum('ijk,ljk->il',Core[i],Core[i]) will be identity matrix
    '''
    tt_model_o = tt_model.clone()
    tt_model_o.orthogonalize(site)
    return tt_model_o

