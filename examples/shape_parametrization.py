import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec  

class BasisFunctions:
    def __init__(self, nbFct, nbSeg, nbIn, nbOut, nbDim, data=None):
        self.nbFct = nbFct
        self.nbSeg = nbSeg
        self.nbIn = nbIn
        self.nbOut = nbOut
        self.nbDim = nbDim
        self.data = data
        self.compute_BC()
        

    @staticmethod
    def binomial(n, i):
        if n >= 0 and i >= 0:
            b = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
        else:
            b = 0
        return b

    @staticmethod
    def block_diag(A, B):
        out = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
        out[: A.shape[0], : A.shape[1]] = A
        out[A.shape[0] :, A.shape[1] :] = B
        return out
    
    def compute_BC(self):
        # Your existing code related to BC matrix initialization
        B0 = np.zeros((self.nbFct, self.nbFct))
        for n in range(1, self.nbFct + 1):
            for i in range(1, self.nbFct + 1):
                B0[self.nbFct - i, n - 1] = (
                    (-1) ** (self.nbFct - i - n)
                    * (-self.binomial(self.nbFct - 1, i - 1))
                    * self.binomial(self.nbFct - 1 - (i - 1), self.nbFct - 1 - (n - 1) - (i - 1))
                )
        B = np.kron(np.eye(self.nbSeg), B0)
        C0 = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]]).T
        C0 = self.block_diag(np.eye(self.nbFct - 4), C0)
        C = np.eye(2)
        for n in range(self.nbSeg - 1):
            C = self.block_diag(C, C0)
        C = self.block_diag(C, np.zeros((self.nbFct - 2, 1)))
        C[-1, 0] = np.array([1])
        C[-1, -1] = np.array([0])
        C[-2, 0] = np.array([0])
        C[-2, -1] = np.array([1])
        self.C = C
        self.BC = B @ C
        self.M = np.kron(self.BC, self.BC)

    def computePsiList1D(self, t):
        T = np.zeros((1, self.nbFct))
        dT = np.zeros((1, self.nbFct))
        phi = np.zeros((len(t), self.BC.shape[1]))
        dphi = np.zeros_like(phi)
        for k in range(0, len(t)):
            tt = np.mod(t[k], 1 / self.nbSeg) * self.nbSeg
            id = np.round(t[k] * self.nbSeg - tt)
            if id < 0:
                tt = tt + id
                id = 0
            if id > (self.nbSeg - 1):
                tt = tt + id - (self.nbSeg - 1)
                id = self.nbSeg - 1
            p1 = np.linspace(0, self.nbFct - 1, self.nbFct)
            p2 = np.linspace(0, self.nbFct - 2, self.nbFct - 1)
            T[0, :] = tt**p1
            dT[0, 1:] = p1[1:] * tt**p2 * self.nbSeg
            idl = id * self.nbFct + p1
            idl = idl.astype("int")
            phi[k, :] = T @ self.BC[idl, :]
            dphi[k, :] = dT @ self.BC[idl, :]
        Psi = np.kron(phi, np.eye(self.nbOut))
        dPsi = np.kron(dphi, np.eye(self.nbOut))
        return Psi, dPsi, phi
    
    def compute_w(self, data, t):

        data_pos = data[:, 0:2]
         # Resampling 
        N = data_pos.shape[0]
        # x = np.linspace(1, N, N)
        x =  (N-1)/(data[:, 0][-1] - data[:, 0][0]) * (data_pos[:, 0] - data_pos[:, 0][0])+1
        xi = np.linspace(1, N, self.nbDim)
        x0 = np.empty((self.nbDim, 2))
        x0[:, 1] = np.interp(xi, x, data_pos[:, 1])

        # Select value dimension only
        x0 = x0[:, 1]
        self.x0 = x0
        Psi, dPsi, phi = self.computePsiList1D(t)

        # Batch estimation of superposition weights from permuted data
        w_b = np.linalg.pinv(Psi) @ self.x0
        w_b_vis = np.kron(self.C, np.eye(self.nbOut)) @ w_b  # for visualization
        self.w_b = w_b
        self.w_b_vis = w_b_vis


# Load and generate data
# ===============================
import sys
sys.path.append('./../')
data = np.load(sys.path[0]+"/meshes/mustared_points.npy", allow_pickle="True")
# data = np.load(sys.path[0]+"/meshes/sugar_points.npy", allow_pickle="True")
# data = np.load(sys.path[0]+"/meshes/square_points.npy", allow_pickle="True")

BF_ys = BasisFunctions(nbFct=4, nbSeg=30, nbIn=1, nbOut=1, nbDim=4*30*10, data=data)

# compute weights
t = np.linspace(0, 1 - 1 / BF_ys.nbDim, BF_ys.nbDim)
BF_ys.compute_w(data, t)

Psi, dPsi, phi = BF_ys.computePsiList1D(t)
# # Reconstruction of trajectories
x_b = Psi @ BF_ys.w_b
# dx_b = dPsi @ BF_ys.w_b

# Visualization
# ===============================

fig = plt.figure(figsize=(10, 5))
gs = GridSpec(3, 2, figure=fig)
ax0 = fig.add_subplot(321)
ax1 = fig.add_subplot(322)
ax2 = fig.add_subplot(gs[1:, :], polar=True)

ax0.set_title("Reference radius")

ax0.plot(t*np.pi*2, BF_ys.x0, linewidth=4, c=[0.8, 0.6, 0.6])
ax0.set_ylabel("radius/m")
ax0.set_xlabel("radian")



ax1.set_title("Reconstructed radius")
ax1.plot(t, BF_ys.x0, linewidth=4, c=[0.8, 0.6, 0.6])
ax1.plot(t, x_b, linewidth=4, c=[0.6, 0.6, 0.6])
t_seg = np.zeros(BF_ys.nbSeg * BF_ys.nbFct)
for n in range(BF_ys.nbSeg):
    ax1.axvline(x=n/BF_ys.nbSeg, ls='--', lw=1)
    t_seg_n = np.linspace(0, 1 / BF_ys.nbSeg, BF_ys.nbFct) + (n) / BF_ys.nbSeg
    t_seg[BF_ys.nbFct * n : BF_ys.nbFct * (n + 1)] = t_seg_n
ax1.axvline(x=1.0, ls='--', lw=1)
ax1.plot(t_seg, BF_ys.w_b_vis, "o-")
short_t = [0, 1/(np.pi*2), 2/(np.pi*2), 3/(np.pi*2), 4/(np.pi*2), 5/(np.pi*2), 6/(np.pi*2)]
modified_x_labels =  [f"{x * 2*np.pi:.0f}" for x in short_t]
ax1.set_xticks(short_t)
ax1.set_xticklabels(modified_x_labels)

ax1.set_ylabel("radius/m")
ax1.set_xlabel("radian")

#draw polar plot
theta = np.array([t]).T*(np.pi*2) - np.pi
ax2.plot(theta, BF_ys.x0, "o-")
ax2.plot(theta, x_b, "o-")
ax2.set_rmax(0.15)
ax2.set_rticks([0.05, 0.1, 0.15])  # Less radial ticks
ax2.set_rlabel_position(-22.5)  # Move radial labels
plt.show()


