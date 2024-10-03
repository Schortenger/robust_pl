import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import celluloid
from celluloid import Camera



def plot_chain(joint_loc, link_lengths, x_obst=[], r_obst=[], x_target=[], rect_patch=[], 
    batch=False, skip_frame=1, title=None, save_as=None,figsize=3,
    color_intensity=0.9, alpha=0.5, contrast=0.4, idx_highlight=[], lw=5):

    fig = plt.figure(edgecolor=[0.1,0.1,0.1])

    fig.set_size_inches(figsize, figsize)



    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    xmax_ = 1.1*np.sum(link_lengths)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-xmax_, xmax_), ylim=(-xmax_, xmax_))

    for i,x in enumerate(x_obst):
        circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
        ax.add_patch(circ)
    for i, x_ in enumerate(rect_patch):
        rect = plt.Rectangle(rect_patch[i][0:2],rect_patch[i][2],rect_patch[i][3], color='c',alpha=0.5)
        ax.add_patch(rect)
    color_ = ['r','g']
    for i, x_ in enumerate(x_target):
        ax.plot(x_[0],x_[1],'o'+color_[i], markersize=5)

    ax.legend(["target","obstacle"])

    T = joint_loc.shape[0]
    idx = np.arange(0,int(T), skip_frame)
    k_ = np.linspace(0.3,0.7,len(idx))[::-1]
    k_[0]=1
    for count,i in enumerate(idx):
        # color_ = np.where(motion, 1-k_[count], contrast)
        x = joint_loc[i,:,0]
        y = joint_loc[i,:,1]
        plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='g',lw=lw,mfc='w',
                    solid_capstyle='round', alpha= k_[count])
        plt.plot(joint_loc[i,-1,0],joint_loc[i,-1,1],'oy', markersize=3)


    for count,i in enumerate(idx_highlight):
        color_ = [0.1]*3
        x = joint_loc[i,:,0]
        y = joint_loc[i,:,1]
        plt.plot(x, y, 'o-',zorder=0.9,marker='o',color='k',lw=lw,mfc='w',
                    solid_capstyle='round', alpha=0.5)

    plt.plot(0,0,color='y',marker='o', markersize=15)
    plt.grid(True)

    if not title is None:
        plt.title(title)
    if not save_as is None:
        fig.savefig('./images/'+save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt



###########################################################################
###########################################################################

def plot_point_mass(x_t, x_target, xmax=1, x_obst=[],r_obst=[],batch=False,title=None, save_as=None,figsize=3):

    fig = plt.figure(edgecolor=[0.1,0.1,0.1])
    fig.set_size_inches(figsize, figsize)

    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-xmax, xmax), ylim=(-xmax, xmax))

    if not x_obst is None:
        for i,x in enumerate(x_obst):
            circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
            ax.add_patch(circ)
    # if not rect_patch is None:
    #     rect = plt.Rectangle(rect_patch[0:2],rect_patch[2],rect_patch[3], color='c',alpha=0.5)
    #     ax.add_patch(rect)
    
    
    ax.plot(x_t[:,0,0],x_t[:,0,1],'og', markersize=10)
    

    for i in range(x_t.shape[0]):
        plt.plot(x_t[i,:,0],x_t[i,:,1],'-b',label='_nolegend_')
    
    ax.plot(x_target[0],x_target[1],'or', markersize=10)
  
#     ax.set_ylabel('y',fontsize=10)
    ax.legend(["init","target","obstacle",])
#     plt.grid("True")
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    if not title is None:
        plt.title(title)
    if not save_as is None:
        fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt

###########################################################################
###########################################################################

def plot_car(x_t, x_target, xmax=1, x_obst=[],r_obst=[],
                batch=False,title=None, save_as=None,figsize=3,
                scale=0, skip=20):
    # x_t: (x,y,theta)

    fig = plt.figure(edgecolor=[0.1,0.1,0.1])
    fig.set_size_inches(figsize, figsize)

    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-xmax, xmax), ylim=(-xmax, xmax))

    if not x_obst is None:
        for i,x in enumerate(x_obst):
            circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
            ax.add_patch(circ)
    # if not rect_patch is None:
    #     rect = plt.Rectangle(rect_patch[0:2],rect_patch[2],rect_patch[3], color='c',alpha=0.5)
    #     ax.add_patch(rect)
    

    ax.plot(x_t[:,0,0],x_t[:,0,1],'og', markersize=10)
    ax.plot(x_target[0],x_target[1],'or', markersize=10)
    idx = np.arange(0,x_t.shape[1],skip)
    for i in range(x_t.shape[0]):
        plt.plot(x_t[i,:,0],x_t[i,:,1],'--b')
        plt.quiver(x_t[i,idx,0], x_t[i,idx,1],np.cos(x_t[i,idx,2]),np.sin(x_t[i,idx,2]), scale=scale)



    ax.legend(["target","init","obstacle"])
    plt.grid("True")

    if not title is None:
        plt.title(title)
    if not save_as is None:
        fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt

###########################################################################
###########################################################################

def plt_pendulum(theta_t, dt=0.01, title=None, save_as=None,figsize=3, scale=1, skip=10, animation=False, name='single_pendulum.mp4'):
    # theta_t:  T x 1

    fig = plt.figure(edgecolor=[0.1,0.1,0.1])
    fig.set_size_inches(figsize, figsize)

    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-1, 1.), ylim=(-1., 1.))
    theta_t =  theta_t.reshape(-1)
    T = theta_t.shape[0]
    idx = np.arange(T)
    y_t = -np.cos(theta_t)
    x_t = np.sin(theta_t)
    xy_t = np.concatenate((x_t.reshape(-1,1),y_t.reshape(-1,1)),axis=-1)
    plt.plot(0,0,"ok", markersize=1*scale)
    
    camera = Camera(fig)
    if animation:
        dt=dt*skip
        interval = (1/dt)*10**-3 # in ms

    for i in range(0,T,skip):
        alpha_ = np.clip(0.1+i*1.0/T, 0,1) if (not animation) else 1.0
        plt.plot([0,xy_t[i,0]],[0,xy_t[i,1]],'-b', linewidth=0.25*scale, alpha=alpha_)
        plt.plot(0.,0.,"ob", markersize=0.1*scale, alpha=alpha_)
#         plt.plot(xy_t[i,0],xy_t[i,1],"ob", markersize=scale, alpha=alpha_)       
        if animation:
            camera.snap()
            
    if animation:
        animation = camera.animate(interval=interval, repeat=False)
        animation.save(name)




    # ax.legend(["target","init","obstacle"])
    plt.grid("True")

    if not title is None:
        plt.title(title)
    if not save_as is None:
        fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt

###########################################################################
###########################################################################

def plt_cartpendulum(x_t, theta_t, length=1.0, dt=0.001, title=None, save_as=None,
                    figsize=3, scale=1, skip=10, file_name='cartpole', animation=False):
    # theta_t:  T x 1

    fig = plt.figure(edgecolor=[0.1,0.1,0.1])
    fig.set_size_inches(figsize, figsize)

    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    x_t = x_t.reshape(-1)
    x_max = np.max(np.abs(x_t))+0.1;
    xlim = (-x_max-1, x_max+1)
    ylim=(-2*length, 2*length)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=xlim, ylim=ylim)
    theta_t = theta_t.reshape(-1)
    T = theta_t.shape[0]
    idx = np.arange(T)
    cy_t = -length*np.cos(theta_t)
    cx_t = x_t+length*np.sin(theta_t)
    cxy_t = np.concatenate((cx_t.reshape(-1,1),cy_t.reshape(-1,1)),axis=-1)
    h = length/3; w = 2*h
    
    camera = Camera(fig)
    if animation:
        interval = skip*dt*10**3
    for i in range(0,T,skip):
        alpha_ = np.clip(0.2+i*1.0/T, 0,1) if (not animation) else 1.0
        ax.add_patch(patches.Rectangle((x_t[i]-w/2,-h), w, h,fill=True, color='blue', alpha=alpha_))
        ax.plot(x_t[i],0,"or", markersize=0.1*scale, alpha=alpha_)
        ax.plot([x_t[i],cxy_t[i,0]],[0,cxy_t[i,1]],'-r', linewidth=scale, alpha=alpha_)
        ax.plot(cxy_t[i,0],cxy_t[i,1],"or", markersize=scale, alpha=alpha_)
        if animation:
            camera.snap()
        alpha_ = 1.0
    ax.add_patch(patches.Rectangle((x_t[i]-w/2,-h), w, h,fill=True, color='green', alpha=alpha_))
    ax.plot(x_t[-1],0,"or", markersize=0.1*scale, alpha=alpha_)
    ax.plot([x_t[-1],cxy_t[-1,0]],[0,cxy_t[-1,1]],'-r', linewidth=scale, alpha=alpha_)
    ax.plot(cxy_t[-1,0],cxy_t[-1,1],"or", markersize=scale, alpha=alpha_)   
    plt.grid("True")

            
    if animation:
        animation = camera.animate(interval=interval, repeat=False)
        animation.save(file_name+'.mp4')

        if not title is None:
            plt.title(title)
#         if not save_as is None:
#             fig.savefig(save_a+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt


###########################################################################
###########################################################################

def plt_double_pendulum(theta_t, dt=0.01, length=1.0, title=None, save_as=None,figsize=3, scale=1, skip=10, animation=False):
    # theta_t:  T x 1

    fig = plt.figure(edgecolor=[0.1,0.1,0.1])
    fig.set_size_inches(figsize, figsize)

    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-2.1*length, 2.1*length), ylim=(-2.1*length, 2.1*length))
    theta_1_t = theta_t[:,0].reshape(-1)
    theta_2_t = theta_t[:,1].reshape(-1)
    T = theta_t.shape[0]
    idx = np.arange(T)
    y_1_t = -np.cos(theta_1_t)
    x_1_t = np.sin(theta_1_t)
    y_2_t = y_1_t-np.cos(theta_2_t+theta_1_t)
    x_2_t = x_1_t+np.sin(theta_2_t+theta_1_t)

    xy_1_t = np.concatenate((x_1_t.reshape(-1,1),y_1_t.reshape(-1,1)),axis=-1)
    xy_2_t = np.concatenate((x_2_t.reshape(-1,1),y_2_t.reshape(-1,1)),axis=-1)
    plt.plot(0,0,"ok", markersize=1*scale)
    
    camera = Camera(fig)
    if animation:
        skip=1

    for i in range(0,T,skip):
        alpha_ = np.clip(0.1+i*1.0/T, 0,1) if (not animation) else 1.0
        plt.plot([0,xy_1_t[i,0]],[0,xy_1_t[i,1]],'-b', linewidth=0.25*scale, alpha=alpha_)
        plt.plot([xy_1_t[i,0],xy_2_t[i,0]],[xy_1_t[i,1],xy_2_t[i,1]],'-b', linewidth=0.25*scale, alpha=alpha_)
        plt.plot(0.,0.,"ob", markersize=0.1*scale, alpha=alpha_)
#         plt.plot(xy_t[i,0],xy_t[i,1],"ob", markersize=scale, alpha=alpha_)       
        if animation:
            camera.snap()
            
    if animation:
        animation = camera.animate(interval=10, repeat=False)
        animation.save('double_pendulum.mp4')

    # ax.legend(["target","init","obstacle"])
    plt.grid("True")

    if not title is None:
        plt.title(title)
    if not save_as is None:
        fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt


###########################################################################
###########################################################################
def R_func(x):
    R_Matrix = [[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]
    return np.array(R_Matrix)


def plot_planarpush_general(x_t, u_t, x_target, step_skip=1, dt=0.01, xmax=1, x_obst=[],r_obst=[],batch=False,title=None, save_as=None,figsize=3, scale=0, slider_r=0.06, animation=False, data=None, file_name='planar_push'):
    # x_t: (x,y,theta, px, py)
    fig = plt.figure(edgecolor=[0.1,0.1,0.1])
    fig.set_size_inches(figsize, figsize)

    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-xmax, xmax), ylim=(-xmax, xmax))

    if not x_obst is None:
        for i,x in enumerate(x_obst):
            circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
            ax.add_patch(circ)

    # color_list = ['y', 'g', 'b', 'm', 'orange', 'r', 'k', 'c', 'bisque', 'blueviolet', 'brown', 'darkblue', ]
    import matplotlib.colors as colors
    color_list = list(colors._colors_full_map.values())
    print()
    cx = x_t[:, :, 0] #bs x T
    cy = x_t[:, :, 1]
    theta = x_t[:, :, 2]
    px = x_t[:, :, 3]
    py = x_t[:, :, 4]
    R = R_func(theta).transpose(2, 3, 0,1) # bsxTx2x2
    #slider plot
    # rec = np.array([[-slider_r, -slider_r, slider_r, slider_r, -slider_r], [-slider_r, slider_r, slider_r, -slider_r, -slider_r]])
    # msh = np.einsum('ktij,jl -> ktil', R, rec) + np.repeat(np.stack([cx, cy], axis=2)[:, :, :, None], 5, axis=3) # bsxTx2x5

    #object contour
    data_pos = data[:, 0:2]
    obj_theta = data_pos[:, 0]
    obj_radius = data_pos[:, 1]

    obj_x = obj_radius * np.cos(obj_theta)
    obj_y = obj_radius * np.sin(obj_theta)
    obj_points = np.stack([obj_x, obj_y], axis=0)
    msh = np.einsum('ktij,jl -> ktil', R, obj_points) + np.repeat(np.stack([cx, cy], axis=2)[:, :, :, None], obj_points.shape[-1], axis=3) # bsxTx2x5


    plt_msh = msh.transpose(2, 3, 0, 1) #2x5xbsxT

    camera = Camera(fig)
    if animation:
        dt=dt*step_skip
        interval = (1/dt)#*10**-3 # in ms

    #pusher plot
    #face switching
    #u_t: bs x T x 3
    # face_angle = (u_t[:, :, -1]*np.pi/2)# bsxT
    # R_face = R_func(theta[:, :-1]+face_angle).transpose(2, 3, 0,1) # bsxTx2x2
    # pusher_xys_rel = np.einsum('ktil, ktlj -> ktij',R_face, x_t[:, :-1, 3:5][:, :, :, None])  #bsxTx2x1
    pusher_xys_rel = np.einsum('ktil, ktlj -> ktij',R[:, :-1], x_t[:, :-1, 3:5][:, :, :, None])  #bsxTx2x1
    # pusher_xys_rel = x_t[:, :-1, 3:5][:, :, :, None]  #bsxTx2x1
    pusher_xys = pusher_xys_rel + x_t[:, :-1, :2][:, :, :, None] #bsxTx2x1



    quil = 0.1 #quiver length

    T = plt_msh.shape[3]
    if animation: # for loop takes long time
        for bt in range(len(plt_msh[0][0])):
            for i in range(0,T-1,step_skip):
                # print("plot_step", i)
                alpha_ = np.clip(0.1+i*1.0/T, 0,1) if (not animation) else 1.0
                # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=((random.random(),random.random(),random.random(), 1)))
                plt.plot(plt_msh[0,:,bt,i], plt_msh[1,:,bt, i], color=color_list[bt], alpha=alpha_)
                ax.plot(cx[bt, i], cy[bt, i],'og', markersize=10)
                ax.quiver(cx[bt, i], cy[bt, i], quil*np.cos(theta[bt, i]), quil*np.sin(theta[bt, i]), color='g', scale=0.5, width=0.01)
                # plt.plot(plt_msh[0,0:2,bt, i], plt_msh[1,0:2,bt, i], 'black', linewidth=3, alpha=alpha_)
                ax.scatter(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50, alpha=alpha_)
                camera.snap()
    plt.grid("True")
    
    # else: #not animated, plot in batch form
    if animation:
        fig2 = plt.figure(edgecolor=[0.1,0.1,0.1])
        fig2.set_size_inches(figsize, figsize)
        ax2 = fig2.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-xmax, xmax), ylim=(-xmax, xmax))
        for bt in range(len(plt_msh[0][0])):
            # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
            # plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
            # ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
            ax2.plot(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], color=color_list[bt])
            # ax2.plot(plt_msh[0,0:2,bt, -1], plt_msh[1,0:2,bt, -1], 'black', linewidth=3)
            ax2.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   

    else:
        for bt in range(len(plt_msh[0][0])):
            # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
            # plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
            # ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
            color_rgb = color_list[bt]
            ax.plot(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], color=color_rgb)
            # ax.plot(plt_msh[0,0:2,bt, -1], plt_msh[1,0:2,bt, -1], 'black', linewidth=3)
            ax.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_rgb, edgecolors='r', marker='o', s=50)   

    # ax.plot(x_t[:,0,0],x_t[:,0,1],'og', markersize=10)
    ax.plot(x_target[0],x_target[1],'or', markersize=10)
    

    ax.quiver(x_target[0], x_target[1], quil*np.cos(x_target[2]), quil*np.sin(x_target[2]),linestyle='dashed', color='r', scale=0.5, width=0.01)

            
    if animation:
        # animation = camera.animate(interval=interval, repeat=False)
        animation = camera.animate()
        animation.save(file_name + '.mp4')


    # ax.legend(["target","init","obstacle"])
    plt.grid("True")

    if not title is None:
        plt.title(title)
    if not save_as is None:
        fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt


def plot_planarpush(x_t, u_t, x_target, step_skip=1, dt=0.01, xmax=1, x_obst=[],r_obst=[],batch=False,title=None, save_as=None,figsize=3, scale=0, slider_r=0.06, animation=False):
    # x_t: (x,y,theta, px, py)
    fig = plt.figure(edgecolor=[0.1,0.1,0.1])
    fig.set_size_inches(figsize, figsize)

    # fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.9)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-xmax, xmax), ylim=(-xmax, xmax))

    if not x_obst is None:
        for i,x in enumerate(x_obst):
            circ = plt.Circle(x,r_obst[i],color='grey',alpha=0.5)
            ax.add_patch(circ)
    # if not rect_patch is None:
    #     rect = plt.Rectangle(rect_patch[0:2],rect_patch[2],rect_patch[3], color='c',alpha=0.5)
    #     ax.add_patch(rect)
    
    # cx = x_t[:, 0, 0]
    # cy = x_t[:, 0, 1]
    # theta = x_t[:, 0, 2]
    # px = x_t[:, 0, 3]
    # py = x_t[:, 0, 4]
    # R = R_func(theta).transpose(2, 0, 1) # bsx2x2
    # rec = np.array([[-slider_r, -slider_r, slider_r, slider_r, -slider_r], [-slider_r, slider_r, slider_r, -slider_r, -slider_r]])
    # msh = np.einsum('kij,jl -> kil', R, rec) + np.repeat(np.stack([cx, cy], axis=1)[:, :, None], 5, axis=2) # bsx2x5
    # plt_msh = msh.transpose(1, 2, 0) #2x5xbs
    # plt.plot(plt_msh[0,:,:], plt_msh[1,:,:], 'darkgray')
    # plt.plot(plt_msh[0,0:2,:], plt_msh[1,0:2,:], 'black')

    # #pusher plot
    # pusher_xys_rel = np.einsum('kil, klj -> kij',R, x_t[:, 0, 3:5][:, :, None])  #bsx2x1
    # pusher_xys = pusher_xys_rel + x_t[:, 0, :2][:, :, None] #bsx1x2
    # ax.plot(pusher_xys[:, 0, 0], pusher_xys[:, 1, 0], 'og',markersize=4)

    # color_list = ['y', 'g', 'b', 'm', 'orange', 'r', 'k', 'c', 'bisque', 'blueviolet', 'brown', 'darkblue', ]
    import matplotlib.colors as colors
    color_list = list(colors._colors_full_map.values())
    print()
    cx = x_t[:, :, 0] #bs x T
    cy = x_t[:, :, 1]
    theta = x_t[:, :, 2]
    px = x_t[:, :, 3]
    py = x_t[:, :, 4]
    R = R_func(theta).transpose(2, 3, 0,1) # bsxTx2x2
    #slider plot
    rec = np.array([[-slider_r, -slider_r, slider_r, slider_r, -slider_r], [-slider_r, slider_r, slider_r, -slider_r, -slider_r]])
    msh = np.einsum('ktij,jl -> ktil', R, rec) + np.repeat(np.stack([cx, cy], axis=2)[:, :, :, None], 5, axis=3) # bsxTx2x5
    plt_msh = msh.transpose(2, 3, 0, 1) #2x5xbsxT

    camera = Camera(fig)
    if animation:
        dt=dt*step_skip
        interval = (1/dt)#*10**-3 # in ms


    #pusher plot
    #face switching
    #u_t: bs x T x 3
    face_angle = (u_t[:, :, 0]*np.pi/2)# bsxT
    R_face = R_func(theta[:, :-1]+face_angle).transpose(2, 3, 0,1) # bsxTx2x2

    pusher_xys_rel = np.einsum('ktil, ktlj -> ktij',R_face, x_t[:, :-1, 3:5][:, :, :, None])  #bsxTx2x1
    pusher_xys = pusher_xys_rel + x_t[:, :-1, :2][:, :, :, None] #bsxTx2x1

    T = plt_msh.shape[3]
    if animation: # for loop takes long time
        for bt in range(len(plt_msh[0][0])):
            for i in range(0,T-1,step_skip):
                # print("plot_step", i)
                alpha_ = np.clip(0.1+i*1.0/T, 0,1) if (not animation) else 1.0
                # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=((random.random(),random.random(),random.random(), 1)))
                plt.plot(plt_msh[0,:,bt,i], plt_msh[1,:,bt, i], color=color_list[bt], alpha=alpha_)
                plt.plot(plt_msh[0,0:2,bt, i], plt_msh[1,0:2,bt, i], 'black', linewidth=3, alpha=alpha_)
                # ax.scatter(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50, alpha=alpha_)
                camera.snap()
    plt.grid("True")
    
    # else: #not animated, plot in batch form
    if animation:
        fig2 = plt.figure(edgecolor=[0.1,0.1,0.1])
        fig2.set_size_inches(figsize, figsize)
        ax2 = fig2.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-xmax, xmax), ylim=(-xmax, xmax))
        for bt in range(len(plt_msh[0][0])):
            # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
            # plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
            # ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
            ax2.plot(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], color=color_list[bt])
            ax2.plot(plt_msh[0,0:2,bt, -1], plt_msh[1,0:2,bt, -1], 'black', linewidth=3)
            # ax2.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   

    else:
        for bt in range(len(plt_msh[0][0])):
            # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=color_list[bt])
            # plt.plot(plt_msh[0,0:2,bt, ::step_skip], plt_msh[1,0:2,bt, ::step_skip], 'black', linewidth=3)
            # ax.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c=color_list[bt], edgecolors='r', marker='o', s=50)   
            color_rgb = color_list[bt]
            ax.plot(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], color=color_rgb)
            ax.plot(plt_msh[0,0:2,bt, -1], plt_msh[1,0:2,bt, -1], 'black', linewidth=3)
            # ax.scatter(pusher_xys[bt, -1, 0, 0], pusher_xys[bt, -1, 1, 0], c=color_rgb, edgecolors='r', marker='o', s=50)   

    # ax.plot(x_t[:,0,0],x_t[:,0,1],'og', markersize=10)
    ax.plot(x_target[0],x_target[1],'or', markersize=10)
    
    quil = 0.1 #quiver length
    ax.quiver(x_target[0], x_target[1], quil*np.cos(x_target[2]), quil*np.sin(x_target[2]), color='r', scale=0.5, width=0.01)

            
    if animation:
        # animation = camera.animate(interval=interval, repeat=False)
        animation = camera.animate()
        animation.save('planar_push.mp4')


    # ax.legend(["target","init","obstacle"])
    plt.grid("True")

    if not title is None:
        plt.title(title)
    if not save_as is None:
        fig.savefig(save_as+".jpeg",bbox_inches='tight', pad_inches=0.01, dpi=300)

    return plt

###########################################################################
###########################################################################
from matplotlib.ticker import LinearLocator
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm

def plot_contour(x_t,ttdp, data=None, contour_scale=50, 
            figsize=10, markersize=3,log_norm=False, device="cpu"):
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-white')
    x = torch.linspace(-1,1,100)
    y = 1*x
    Z = torch.empty((len(x),len(y)))
    X, Y = torch.meshgrid(x, y)
    print(X.shape, Y.shape)
    XY = torch.concat((X.reshape(-1,1),Y.reshape(-1,1)),dim=-1).to(device)
    action = torch.tensor([0.,0.]).to(device).view(1,-1).expand(XY.shape[0],-1)
    Z = (ttdp.get_value(ttdp.v_model,XY) + ttdp.reward_normalized(XY,action)).reshape(X.shape[0],X.shape[1])
    cmap = 'binary_r'    
    if log_norm == True:
        levels = 10.**(0.25*np.arange(-6,14))
        cs = plt.contour(X.to('cpu').numpy(), Y.to('cpu').numpy(), 
            Z.to('cpu').numpy(), contour_scale, cmap=cmap, shade=True,
            locator=ticker.LogLocator(),
            levels=levels, norm=LogNorm(), alpha=1)
    else:
        cs = plt.contour(X.to('cpu').numpy(), Y.to('cpu').numpy(),
                 Z.to('cpu').numpy(), contour_scale, 
                 cmap=cmap, shade=True,alpha=1);

    plt.colorbar(cs);
    
    plt.plot(x_t[:,0,0],x_t[:,0,1],'og', markersize=10)
    for i in range(x_t.shape[0]):
        plt.plot(x_t[i,:,0],x_t[i,:,1],'*b')

#     if not (data is None):
#         plt.plot(data[:,0],data[:,1],'ob', markersize=markersize)
    plt.rcParams["figure.figsize"] = (figsize, figsize)
    plt.axis('square')
    return plt

