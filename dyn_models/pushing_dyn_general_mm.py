'''
    General pushing model, different contact modes (sticking, sliding up, sliding down, separation and
    face-switching mechanism are formulated in the forward dynamics function as
    conditions. 
    This version eliminates the limitation on object shape, and number of contact points.

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
        self.Dx = 5 # 7 #[slider_x, slider_y, slider_theta, pusher_x, pusher_y, pusher_xdot, pusher_ydot]
        self.Du = 4 # [fn, ft, vn, vt]
        self.slider_r = torch.tensor(SLIDER_R).to(device)
        self.pusher_r = torch.tensor(PUSHER_R).to(device)
        self.pusher_hheight = 0.12 #higher height to switch face
        self.contact_tol = torch.tensor(CONTACT_TOL).to(device)  # Tolerance for contact evaluation
        self.px = (-self.slider_r - self.pusher_r).to(device)
        self.u_ps = torch.tensor(0.3).to(device)  # Friction coefficient between pusher and slider
        self.u_gs = torch.tensor(0.35).to(device)  # Friction  coefficient between ground and slider
        self.m = torch.tensor(0.5).to(device)  # Mass of the slider
        self.g = torch.tensor(9.8).to(device)  # Gravity

        val, _ = integrate.dblquad(lambda x, y: np.sqrt(x**2 + y**2), 0, self.slider_r, 0, self.slider_r)
        
        self.c = (val / pow(self.slider_r, 2))
        self.device = device
        self.p_target = p_target

    def R_func_2D(self, x):
        R_Matrix = torch.stack([torch.cat([torch.cos(x), -torch.sin(x)], dim=1), 
                            torch.cat([torch.sin(x), torch.cos(x)], dim=1)], dim=1).to(self.device) #bs x 2 x 2
        return R_Matrix
    
    def R_func_3D(self, x):
        R_Matrix = torch.stack([torch.cat([torch.cos(x), -torch.sin(x), torch.zeros_like(x)], dim=1), 
                        torch.cat([torch.sin(x), torch.cos(x), torch.zeros_like(x)], dim=1),  
                        torch.cat([torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x)], dim=1)], dim=1).to(self.device) #bs x 3 x 3
        return R_Matrix

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
    
    def sdfBox(self, point, center=torch.tensor([0, 0]), dimensions=torch.tensor([0.4, 0.2])):
        d = abs(point - center) - np.array(dimensions) * 0.5
        return torch.linalg.norm(torch.maximum(d, 0.0)) + min(torch.max(d), 0.0)
    
    
    def box_edge_function(self, psi, center=torch.tensor([0, 0]), dimensions=torch.tensor([0.12, 0.12])):
        """
        Represent the edge of a rectangle with a function y = f(x).

        Args:
        - x: float, the x-coordinate of the point

        Returns:
        - float: the distance of the point on the edge to origin
        """
        # Define the range of x-values for the rectangle
        x_min = -dimensions[0]
        x_max = dimensions[0]

        # Define the range of y-values for the rectangle
        y_min = -dimensions[1]
        y_max = dimensions[1]

        diag_angle = torch.arctan(dimensions[1]/dimensions[0])

        right_flag = (psi >= -diag_angle) * (psi <= diag_angle)
        top_flag = (psi > diag_angle) * (psi < np.pi - diag_angle)
        left_flag= (psi <= -np.pi + diag_angle) * (psi >= -np.pi) + (psi >= np.pi - diag_angle) * (psi <= np.pi)
        bottom_flag = (psi < -diag_angle) *  (psi > -np.pi + diag_angle)


        y = top_flag * y_max + bottom_flag * y_min + left_flag * (x_min * torch.tan(psi)) + right_flag * (x_max * torch.tan(psi))

        x = right_flag * x_max + left_flag * x_min + top_flag * (y_max / (torch.tan(psi)+1e-5)) + bottom_flag * (y_min / (torch.tan(psi)+1e-5))

        radius = torch.linalg.norm(torch.hstack([x, y]), dim=1).view(-1,1)

        #compute rel. angle between contact normal vector and body frame
        beta = right_flag * torch.tensor(0) + top_flag * torch.pi/2 + left_flag * torch.pi + bottom_flag * (-torch.pi/2)

        return radius, beta.to(self.device)
    

    def get_pxy(self, psi, phi):
        """
        get relative position of pusher w.r.t slider. 
        This function can be used to find the deviation w.r.t. psi and phi, and then contribute to the relative velocity of pusher.        
        """
        radius, beta = self.box_edge_function(psi)
        p_x = (phi + radius)*torch.cos(psi) # pusher x
        p_y = (phi + radius)*torch.sin(psi) # pusher y
        r_c = torch.hstack(p_x, p_y)

        return r_c
    
    def dynamics(self, xs, us):

        """
        param xs: state_param, [slider_x, slider_y, slider_theta, psi(pusher_angle), phi(sdf_distance)], [r, mass, u_gs, u_ps]
        param us: control, [fn, ft, psi_dot, phi_dot]
        """
        s_xy = xs[:,:2]
        s_theta = xs[:,2].view(-1,1)
        psi = xs[:,3].view(-1,1) # relative angle between pusher and slider
        phi = xs[:,4].view(-1,1) # distance between pusher and slider

        param = xs[:, 5:] # [r, mass, u_gs, u_ps]
        radius = param[:, 0][:, None]
        mass = param[:, 1]
        u_gs = param[:, 2]

        beta = psi.clone()
        
        # p_x = (phi + radius)*torch.cos(psi) # pusher x
        # p_y = (phi + radius)*torch.sin(psi) # pusher y
        p_x = radius*torch.cos(psi) # pusher x, bs x 1
        p_y = radius*torch.sin(psi) # pusher y, bs x 1



        R_Matrix = self.R_func_3D(s_theta) # bs x 3 x 3
        f_max = (u_gs * mass *self.g).view(-1,1)

        f_n = us[:, 0]# normal force
        f_t = us[:, 1] # tangential force
        psi_dot = us[:, 2] # psi_dot
        phi_dot = us[:, 3] # phi_dot

        cond_cone_stick = 1 * torch.logical_and(f_t >= -self.u_ps * f_n, f_t <= self.u_ps * f_n) # bs x 1
        cond_cone_up = 1 * (f_t > self.u_ps * f_n) # bs x 1
        cond_cone_down = 1 * (f_t < -self.u_ps * f_n) # bs x 1


        f_t = torch.clamp(f_t, -self.u_ps * f_max.squeeze(1), self.u_ps*f_max.squeeze(1))

        f_c = torch.stack([-f_n, f_t], axis=1) # bs x 2

        f_c_bf = torch.einsum('bij, bjk->bik', self.R_func_2D(beta), f_c[:, :, None]) # bs x 2 x 1 # contact force in body frame

        c = 0.6 # integration constant, Chavan-Dafle et al., 2018
  

        m_max = c * radius * f_max


        L = torch.diag_embed(torch.cat((f_max**(-1), f_max**(-1), m_max**(-1)), dim=1)).to(self.device)

        Jacobian_c =torch.stack([torch.cat([torch.ones_like(p_y), torch.zeros_like(p_x)], dim=-1), torch.cat([torch.zeros_like(p_y),  torch.ones_like(p_x)], dim=-1), 
                        torch.cat([-p_y, p_x], dim=-1)], dim=1).to(self.device) # bs x3 x 2
        
        # #contact complementary constraints
        cond_contact = (phi <= self.contact_tol).float()

        f_c_bf = f_c_bf * cond_contact[:, None] + torch.zeros_like(f_c_bf)* (1 - cond_contact[:, None]) # if contact, then f_c_bf, otherwise, f_c_bf = 0
        wrench = torch.einsum('bij,bjk->bik', Jacobian_c, f_c_bf).to(self.device) # bs x 3 x 1

        s_dot = torch.einsum('bij, bjk, bkm->bim', R_Matrix, L, wrench).squeeze(2).to(self.device)

        next_psi = us[:, 2][:, None]
        next_phi = us[:, 3][:, None]* (1-cond_contact) + torch.zeros_like(phi)*cond_contact

        xs_next = xs[:, :3] + s_dot * self.dt
        x_next = torch.hstack([xs_next, next_psi, next_phi])

        state_param_next = torch.hstack([x_next, param])

        return state_param_next

 
    def cost_func(self, xs, us, tol=0.01, scale=0.5):
        state = xs[:, :5]
        param = xs[:, 5:]
        pos_error = (torch.linalg.norm(state[:,:2], dim=-1)/(tol*scale))
        # ori_error = (xs[:,2]).abs()/(tol*torch.pi)
        ori_error = (torch.linalg.norm(state[:,2][:, None], dim=-1)/(tol*torch.pi))

        cost_state =  pos_error*0.3 + ori_error*1

        cost_force = torch.linalg.norm(us[:, :2], dim=-1)
        cost_vel = torch.linalg.norm(state[:, 3:] - us[:, 2:], dim=-1)
        # sdf_cost = torch.linalg.norm(xs[:,4][:, None], dim=-1)
        cost =  cost_state + cost_force*0.01 + cost_vel*0.01 #+ sdf_cost*0.01
        return cost
    