'''
    explicit dynamics version for planar pushing task, where the
    different contact modes (sticking, sliding up, sliding down, separation and
    face-switching mechanism are formulated in the forward dynamics function as
    conditions. This formulation removes the hard constraints required by the
    implicit version.)
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time, os
import pybullet as p
import pybullet_data
from scipy import integrate
import math
import pdb
torch.set_default_dtype(torch.float64)

# from config import *
"""
Configuration
"""
SLIDER_R = .12 / 2
PUSHER_R = .01/2
PUSHER_X = -SLIDER_R - PUSHER_R
PUSHER_Y = 0
SLIDER_INIT = [0, 0., 0.04]
SLIDER_INIT_THETA = 0
# PUSHER_INIT = [1.3*PUSHER_X, 0, 0]#W.R.T slider-centered frame
PUSHER_ORI = [torch.pi, 0, 0]
TABLE_POS = [SLIDER_INIT[0], 0, 0]
CONTACT_TOL = 1E-3*5
DT = 0.01 # stepping time
T = 500
VEL_BOUND = [[0, 0.05], [-0.05, 0.05]]
ACC_BOUND = [[-1, 1], [-1, 1]]

"""
Pusher-slider-system
"""
class pusher_slider_sys():
    def __init__(self, p_target, dt=0.01, device="cpu"):
        self.dt = dt
        self.Dx = 6 # 7 #[slider_x, slider_y, slider_theta, pusher_x, pusher_y, pusher_xdot, pusher_ydot]
        self.Du = 2
        self.slider_r = torch.tensor(SLIDER_R).to(device)
        self.pusher_r = torch.tensor(PUSHER_R).to(device)
        self.pusher_hheight = 0.12 #higher height to switch face
        self.contact_tol = torch.tensor(CONTACT_TOL).to(device)  # Tolerance for contact evaluation
        self.px = (-self.slider_r - self.pusher_r).to(device)
        self.u_ps = 0.3  # Friction coefficient between pusher and slider
        self.u_gs = 0.35  # Friction  coefficient between ground and slider
        val, _ = integrate.dblquad(lambda x, y: np.sqrt(x**2 + y**2), 0, self.slider_r, 0, self.slider_r)
        
        self.c = (val / pow(self.slider_r, 2))
        self.device = device
        self.p_target = p_target

    def R_func(self, x):
        R_Matrix = [[torch.cos(x), -torch.sin(x)], [torch.sin(x), torch.cos(x)]]
        return torch.array(R_Matrix)

    def C_func(self, x):
        C_Matrix = [[torch.cos(x), torch.sin(x)], [-torch.sin(x), torch.cos(x)]]
        return torch.array(C_Matrix)

    def gama_t_func(self, px, py):
        u_ps = self.u_ps
        c = self.c
        gama_t = (u_ps * c ** 2 - px * py + u_ps * px ** 2) / (c ** 2 + py ** 2 - u_ps * px * py)
        return gama_t.view(-1,1)

    def gama_b_func(self, px, py):
        u_ps = self.u_ps
        c = self.c
        gama_b = (-u_ps * c ** 2 - px * py - u_ps * px ** 2) / (c ** 2 + py ** 2 + u_ps * px * py)
        return gama_b.view(-1,1)

    def Q_func(self, px, py):
        c = self.c
        Q1_M = [[c ** 2 + px ** 2, px * py], [px * py, c ** 2 + py ** 2]]
        Q2_M = c ** 2 + px ** 2 + py ** 2
        Q_Matrix = Q1_M / Q2_M
        return np.array(Q_Matrix)

    def b1_func(self, px, py):
        bs = len(py)
        out_ = torch.empty(bs,1,2).to(self.device)
        c = self.c
        out_[:,0,0] = -py / (c ** 2 + px ** 2 + py ** 2)
        out_[:,0,1] = px / (c ** 2 + px ** 2 + py ** 2)
        return out_

    def b2_func(self, px, py, gama_t):
        bs = len(py)
        out_ = torch.zeros(bs,1,2).to(self.device)
        c = self.c
        out_[:,0,0]= (-py+ gama_t.squeeze(-1) * px) / (c ** 2 + px ** 2 + py ** 2)
        return out_

    def b3_func(self, px, py, gama_b):
        bs = len(py)
        out_ = torch.zeros(bs,1,2).to(self.device)
        c = self.c
        out_[:,0,0] = (-py + gama_b.squeeze(-1) * px) / (c ** 2 + px ** 2 + py ** 2)
        return out_

    def dynamics(self, xs, us):
        # faceid: -1 (left), -2 (bottom), 1 (right), 2 (top)
        us = torch.cat((us[:,1:],us[:,0].view(-1,1)),dim=-1)
        faceid = us[:,-1].view(-1,1)
        faceid = (faceid-2.0)*(faceid<2)+(faceid-1.0)**(faceid>1)
        s_xy = xs[:,:2]
        s_theta = xs[:,2].view(-1,1)
        p_x = xs[:,3]; p_y = xs[:,4]
        vn = us[:, 0][:, None]
        vt = us[:, 1][:, None]
        u = 1*us[:, :2].double()
        bs = xs.size(0)
        face_beta = (torch.abs(faceid-(-1))*torch.pi/2).to(self.device) # batch x 1
        R_mat = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
        R_mat[:,0,0] = torch.cos(face_beta).squeeze(dim=-1)
        R_mat[:,-1,-1] = torch.cos(face_beta).squeeze(dim=-1)
        R_mat[:,0,1] = -1*torch.sin(face_beta).squeeze(dim=-1)
        R_mat[:,1,0] = torch.sin(face_beta).squeeze(dim=-1)

        R_theta = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
        R_theta[:,0,0] = torch.cos(s_theta).squeeze(dim=-1)
        R_theta[:,-1,-1] = torch.cos(s_theta).squeeze(dim=-1)
        R_theta[:,0,1] = -1*torch.sin(s_theta).squeeze(dim=-1)
        R_theta[:,1,0] = torch.sin(s_theta).squeeze(dim=-1)

        Q_mat = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
        Q_mat[:,0,0] = self.c**2 + p_x**2
        Q_mat[:,-1,-1] = self.c**2 + p_y**2
        Q_mat[:,0,1] = p_x*p_y
        Q_mat[:,1,0] = p_x*p_y
        Q_mat = Q_mat/(self.c**2+p_x**2+p_y**2)[:, None, None]
        """
        Let's assume xs[3:5] and us are defined in left face frame
        """
        # u = R_mat.T.dot(x[5:7]) # Represent u in left face frame
        # u = x[5:7]
        # x[3: 5] = R_mat.T.dot(x[3: 5]) #convert to left face frame
        gama_t = self.gama_t_func(self.px, p_y).to(self.device) # bs x 1
        gama_b = self.gama_b_func(self.px, p_y).to(self.device) # bs x 1 

        D = torch.zeros(bs,1,2).to(self.device)
        P1 = torch.eye(2).to(self.device)[None,:,:].expand(bs,-1,-1) # bs x 2 x 2
        P2 = torch.cat([P1[:, 0:1],torch.cat([gama_t, torch.zeros_like(gama_t)], axis=-1)[:, None]],axis=1)  # bs x 2 x 2
        P3 = torch.cat([P1[:, 0:1], torch.cat([gama_b, torch.zeros_like(gama_b)], axis=-1)[:, None]],axis=1) # bs x 2 x 2

        c1 = torch.zeros(bs,1,2).to(self.device) # bs x 1 x 2
        c2 = torch.cat([-gama_t, torch.ones_like(gama_t)], axis =-1)[:, None, :].to(self.device) # bs x 1 x 2
        c3 = torch.cat([-gama_b, torch.ones_like(gama_b)], axis=-1)[:, None, :].to(self.device) # bs x 1 x 2
        b1 = self.b1_func(self.px,p_y).view(-1,2)[:, None] # bs x 1 x 2
        b2 = self.b2_func(self.px,p_y, gama_t).view(-1,2)[:, None, :] # bs x 1 x 2
        b3 = self.b3_func(self.px,p_y, gama_b).view(-1,2) [:, None, :] # bs x 1 x 2
        # dyn1 = torch.empty(bs,7,2)
        dyn1 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta, Q_mat.double(), P1),
                          b1, D, c1], axis=1) # bs x 5 x 2
        dyn2 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta,Q_mat.double(), P2.double()),
                          b2, D, c2], axis=1) # bs x 5 x 2
        dyn3 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta,Q_mat.double(), P3.double()),
                          b3, D, c3], axis=1) # bs x 5 x 2
        dyn4 = torch.cat([torch.zeros(bs, 3,2).to(self.device), torch.tile(torch.eye(2)[None].to(self.device), (bs, 1,1))], axis=1)

        cond_cone_stick = 1 * torch.logical_and(vt >= gama_b * vn, vt <= gama_t * vn) # bs x 1
        cond_cone_up = 1 * (vt > gama_t * vn) # bs x 1
        cond_cone_down = 1 * (vt < gama_b * vn) # bs x 1

        cond_touch = (torch.abs(p_x - self.px) <= self.contact_tol)[:, None]

        #Assuming vn is always >0, so that the seperation mode is only determined by the distance between pusher and slider
        cond1 = 1 * torch.logical_and(cond_cone_stick, cond_touch)# bs x 1
        cond2 = 1 * torch.logical_and(cond_cone_up, cond_touch)# bs x 1
        cond3 = 1 * torch.logical_and(cond_cone_down, cond_touch)# bs x 1
        cond4 = 1 - cond1 - cond2 - cond3# bs x 1

        
        # f is the system state w.r.t. global frame, but assuming pushing the initial left face
        f = cond1 * torch.einsum('ijk,ikl -> ijl', dyn1, u[:,:, None])[..., 0] # bs x 5
        f += cond2 * torch.einsum('ijk,ikl -> ijl', dyn2.double(), u[:,:, None])[..., 0] # bs x 5
        f += cond3 * torch.einsum('ijk,ikl -> ijl', dyn3.double(), u[:,:, None])[..., 0] # bs x 5
        f += cond4 * torch.einsum('ijk,ikl -> ijl', dyn4.double(), u[:,:, None])[..., 0] # bs x 5

        # transform the state obtained by assuming left face to current face
        f[:, :2][:, :, None] = torch.einsum('ijk,ikl -> ijl', R_mat, f[:, :2][:, :, None])

        return torch.cat(((xs[:, :5]+f*self.dt), us[:, -1][:, None]), dim=1).to(self.device).double()
    

    def cost_func(self, xs, us, tol=0.01, scale=0.3):
        consider_next_state = True
        if consider_next_state: # the cost function is more informative w.r.t the action
            next_state = self.dynamics(xs,us)
            pos_error_1 = torch.linalg.norm(xs[:,:2]-self.p_target[:,:2], dim=-1)
            ori_error_1 = (xs[:,2]-self.p_target[:,2]).abs()
            pos_error_2 = torch.linalg.norm(next_state[:,:2]-self.p_target[:,:2], dim=-1)
            ori_error_2 = (next_state[:,2]-self.p_target[:,2]).abs()
            pos_error =  0.5*(pos_error_2 + pos_error_1)/(scale*tol)
            ori_error = 0.5*(ori_error_2 + ori_error_1)/(tol*torch.pi)
        else:
            pos_error = torch.linalg.norm(xs[:,:2]-self.p_target[:,:2], dim=-1)/(tol*scale)
            ori_error = (xs[:,2]-self.p_target[:,2]).abs()/(tol*torch.pi)

        cost_state = 0.5*pos_error + 0.5*ori_error
        cost_face_switch = (xs[:, -1] != us[:, 0])
        cost_vel = torch.linalg.norm(us[:, 1:], dim=-1)
        cost = cost_state+cost_face_switch #*(1 + 0.01*cost_face_switch + 0.001*cost_vel)
    
        return cost
