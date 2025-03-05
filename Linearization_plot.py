import time
import sympy as smp
from sympy.tensor.array.expressions import ArraySymbol
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
from pylab import figure, cm
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

#import tikzplotlib
#import tikzplotlib

anfang = time.time()

W_sat = 0.15
m = 0.35
h_k = 1.25
mu = 1
n = 3.5
K_phi = 0.000075
dx = 0.2
dy = 0.2
dz = 0.2

dt = 1e1
steps = 10_000
w_init = 0.19
w_fill = 0.29

WAP = (W_sat+m)/2

dim=[19,19,7]


# Define Symbolic variables
z = smp.symbols('z', real = True)
w= smp.symbols('W ', real = True)
w_ijk, w_ijk1, w_ij1k, w_i1jk = smp.symbols('w_{ijk}, w_{ij(k+1)} w_{i(j+1)k}, w_{(i+1)jk} ',real=True)
w_ijk_1, w_ij_1k, w_i_1jk = smp.symbols('w_{ij(k-1)}, w_{i(j-1)k}, w_{(i-1)jk} ', real =True)


# Define functions and spartial deriviertives
k_w = K_phi * ((w-W_sat)/(m-W_sat))**(3.5)

U = mu*h_k*(-smp.log(abs((w-W_sat)/(m-W_sat))))**(1/n)+z

dudx = (U.subs(w,w_i1jk) - U.subs(w,w_i_1jk))/(2*dx)
#dudx
dudy = (U.subs(w,w_ij1k) - U.subs(w,w_ij_1k))/(2*dy)
#dudy
dudz = (U.subs([(w,w_ijk1),(z,-dz)]) - U.subs([(w,w_ijk_1),(z,dz)]))/(2*dz) #+1
#dudz

dwdx = (w_i1jk - w_i_1jk)/(2*dx)
dwdy = (w_ij1k - w_ij_1k)/(2*dy)
dwdz = (w_ijk1 - w_ijk_1)/(2*dz)

d2udx2 = (U.subs(w,w_i1jk) + U.subs(w,w_i_1jk) - 2*U.subs(w,w_ijk))/(dx**2)
d2udy2 = (U.subs(w,w_ij1k) + U.subs(w,w_ij_1k) - 2*U.subs(w,w_ijk))/(dy**2)
d2udz2 = (U.subs(w,w_ijk1) + U.subs(w,w_ijk_1) - 2*U.subs(w,w_ijk))/(dz**2)

# Symbolic expression of the differenzial equation dW/dt  in continoius time
dwdt = smp.diff(k_w,w)*(dudx*dwdx + dudy*dwdy + dudz*dwdz) - k_w*(d2udx2 + d2udy2 + d2udz2)
dwdt = dwdt.subs(w,w_ijk)
dwdt


# Define Symbolic States
W =ArraySymbol('W', (dim[0], dim[1], dim[2])).as_explicit().as_mutable()


# In[8]:

timer = 0
shape_A = dim[0]*dim[1]*dim[2]
# Hack um Assumptions auf die Variablen Anzuwenden (ohne real= True wirds sonst hässlich)
for X in range(dim[0]):
        for Y in range(dim[1]):
            for Z in range(dim[2]):
                W[X,Y,Z] = smp.Symbol('W_'+str(X)+str(Y)+str(Z),positive=True, real = True )
                timer += 1
                print(str(round((timer/shape_A)*100,1))+"% of something")


# In[9]:


# Extended 3d statespace where outer layer is doubled
Wplus =ArraySymbol("X", (dim[0]+2, dim[1]+2, dim[2]+2)).as_explicit().as_mutable()
Wplus[1:-1,1:-1,1:-1]= W[:,:,:]
Wplus[0,1:-1,1:-1]   = W[0,:,:]
Wplus[-1,1:-1,1:-1]  = W[-1,:,:]
Wplus[1:-1,0,1:-1]   = W[:,0,:]
Wplus[1:-1,-1,1:-1]  = W[:,-1,:]
Wplus[1:-1,1:-1,0]   = W[:,:,0]
Wplus[1:-1,1:-1,-1]  = W[:,:,-1]




# Define Symbolic derivertives
dWdt =ArraySymbol("dWdt", (dim[0], dim[1], dim[2])).as_explicit().as_mutable()


timer = 0
# Set up Statespace symbolically
for X in range(dim[0]):
        for Y in range(dim[1]):
            for Z in range(dim[2]):
                Xplus = X + 1
                Yplus = Y + 1
                Zplus = Z + 1
                dWdt[X,Y,Z] = dwdt.subs([(w_ijk,Wplus[Xplus,Yplus,Zplus]),
                                         (w_ijk1,Wplus[Xplus,Yplus,Zplus+1]),
                                         (w_ij1k,Wplus[Xplus,Yplus+1,Zplus]),
                                         (w_i1jk,Wplus[Xplus+1,Yplus,Zplus]),
                                         (w_ijk_1,Wplus[Xplus,Yplus,Zplus-1]),
                                         (w_ij_1k,Wplus[Xplus,Yplus-1,Zplus]),
                                         (w_i_1jk,Wplus[Xplus-1,Yplus,Zplus]),])
                timer += 1
                print(str(round((timer / shape_A) * 100, 1)) + "% setting up space state symbolically")


# Get a vector as statespace functions
dWdt_vec = dWdt.reshape(int(np.prod(dim)))
dWdt_vec.shape




# Define vector of state variables
W_vec = W.reshape(int(np.prod(dim)))
W_vec.shape




# Compute Derivertive
A_sym = dWdt_vec.diff(W_vec)



timer = 0
# Substitute Opterating point
A_ap = A_sym
timer = 0
for X in range(dim[0]):
        for Y in range(dim[1]):
            for Z in range(dim[2]):
                A_ap = A_ap.subs(W[X,Y,Z],WAP)
                timer += 1
                print(str(round((timer / shape_A) * 100, 1)) + "% subsituting operating point")


# In[16]:


# show A matrix
A_ap


# In[17]:


# As Numpy matrix
A = np.array(A_ap).astype(np.float64)
np.save("A_matrix_14_14_5",A)




# Show Eigenvlaues
#w,v = np.linalg.eig(A)
#w










#
# np.array([1,2,3])
#
#
# # In[49]:
#
#
# F = 0.19*np.ones((dim[0],dim[1],dim[2]))
#
#
#
#
#
# w_max = 0.29
# F[2:3,2:3,0] = w_max
# F[dim[0]-3:dim[0]-2,2:3,0] = w_max
# F[2:3,dim[0]-3:dim[0]-2,0] = w_max
# F[dim[0]-3:dim[0]-2,dim[0]-3:dim[0]-2,0] = w_max
#
#
#
#
#
# #cmap = plt.cm.get_cmap("Blues").copy()
# #cmap.set_bad(color='red')
# #plt.imshow(F[:,:,0], cmap=cmap)
# #plt.title("Watering at {}s".format(t*dt))
# #plt.colorbar()
# #plt.clim(0.19, 0.29)
# #plt.show()
#
#
#
#
#
# x= F.ravel()
#
#
#
#
#
# A_diskret = A*dt
# ende = time.time()
#
# print("vorbereitung hat gedauert:")
# print(ende-anfang)
#
#
# # In[54]:
#
#
#
# steps = 10_000_000
# a = time.time()
# F= np.zeros((dim[0],dim[1],dim[2],steps))
#
#
# #gaussian_irrigate(0)
#
# # Berechnung:
# # x[k+1] = A*x[k] + x[k]
# F[:,:,:,0] = x.reshape(dim[0],dim[1],dim[2])
# for t in range(1,steps,1):
#     x = np.matmul(A_diskret,x) + x
#     F[:,:,:,t] = x.reshape(dim[0],dim[1],dim[2])
#     #print(x)
#     if ((t/steps)*100)%10==0:
#         print("{}% calculated".format(t/steps*100))
#
# e= time.time()
# print(e-a)
#
#
#
# # In[55]:
#
#
# x
#
#
# # In[56]:
#
#
# dim
#
#
# # In[57]:
#
#
#
#
# # In[58]:
#
#
# #for t in range(0,steps,int(steps/10)):
# #    cmap = plt.cm.get_cmap("Blues").copy()
# #    cmap.set_bad(color='red')
# #    plt.imshow(F[:,:,0,t], cmap=cmap)
# #    plt.title("Watering at {}s".format(t*dt))
# #    plt.colorbar()
# #    plt.clim(0.19, 0.29)
# #    plt.show()
#
#
# #
#
# # In[61]:
#
#
# def log_transform(im):
#     '''returns log(image) scaled to the interval [0,1]'''
#     try:
#         (min, max) = (im[im > 0].min(), im.max())
#         if (max > min) and (max > 0):
#             return (np.log(im.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
#     except:
#         pass
#     return im
#
# a = (np.e*0.19-0.29)/(np.e-1)
# b = 1/(0.19-a)
#
# def get_ticks(x):
#     return round(((((np.e)**x)/b+a)*100),1)
#
#
# # In[62]:
#
#
#
# cmap = plt.cm.get_cmap("Blues").copy()
# cmap.set_bad(color='red')
#
# fig = plt.figure(figsize=(20, 20))
#
# grid = AxesGrid(fig, 111,
#                 nrows_ncols=(2, 4),
#                 axes_pad=0.25,
#                 cbar_mode='single',
#                 cbar_location='right',
#                 cbar_pad=0.5
#                 )
#
#
#
# cmap = plt.cm.get_cmap("Blues").copy()
# cmap.set_bad(color='red')
#
# t = 0
# for ax in grid:
#     ax.set_axis_off()
#     im = ax.imshow(log_transform(F[:,:,0,t]), cmap=cmap)#, norm=LogNorm(vmin=0, vmax=1.1))
#     t += int(steps/8)
#
# # when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]
# cbar = ax.cax.colorbar(im)
# cbar = grid.cbar_axes[0].colorbar(im)
#
# ticklabels_list = []
#
# for i in range(6):
#     ticklabels_list.append(str(get_ticks(i/5))+"%")
#
# cbar.ax.set_yticks(np.arange(0, 1.2, 0.2))
# cbar.ax.set_yticklabels(ticklabels_list)
# #tikzplotlib.save("testdatei.tex")
# plt.show()
#
#
# # In[ ]:
# name_datei = "linearized_data" + str(time.time()) + ".np"
# np.save(name_datei,F)
matrix_name = "A_matrix_"+str(dim[0])+"_"+str(dim[1])+"_"+str(dim[2])
np.save(matrix_name, A)