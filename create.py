#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

# In[2]:


a = time.time()

# In[3]:


import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib
import time
from IPython.display import display, HTML

display(HTML("<style>.container { width:99% !important; }</style>"))

# # Formel vom Paper
# $\frac{\delta}{\delta t} W= k'(W) \Big( \frac{\delta}{\delta x} U \frac{\delta}{\delta x} W + \frac{\delta}{\delta y} U \frac{\delta}{\delta y} W + \frac{\delta}{\delta z}U \frac{\delta}{\delta z} W \Big) + k(W)\Big(\frac{\delta^2}{\delta x^2}U + \frac{\delta^2}{\delta y^2}U + \frac{\delta^2}{\delta z^2}U \Big) + f(x,y,z,t)$
#
# # Änderung:
# $\frac{\delta}{\delta t} W= k'(W) \Big( \frac{\delta}{\delta x} U \frac{\delta}{\delta x} W + \frac{\delta}{\delta y} U \frac{\delta}{\delta y} W + \frac{\delta}{\delta z}U \frac{\delta}{\delta z} W \Big) - k(W)\Big(\frac{\delta^2}{\delta x^2}U + \frac{\delta^2}{\delta y^2}U + \frac{\delta^2}{\delta z^2}U \Big) + f(x,y,z,t)$
#
# Probelm: $U(W)$ sinkt mit steigendem W, ein postitives $\frac{\partial^2W}{\partial x ^ 2}$ fürt damit zu einem negativen Wert für  $\frac{\partial^2U}{\partial x ^ 2}$
#
# Jedoch soll bei einem n postitiven $\frac{\partial^2W}{\partial x ^ 2}$ der wert für $W$ steigen, also muss $\frac{\delta}{\delta t} W>0$ sein. Damit feht dort irgendwo ein Minus

# In[4]:


t, mu, h_k, K_phi, x, y, z, W_sat, m, n = smp.symbols('t mu h_k K_phi x y z W_sat m n', real=True)
w, w_ap, dt, mu, h_k, K_phi, dx, dy, dz, W_sat, m, n = smp.symbols(
    'W, W_{AP} Delta_t, mu, h_k, K_phi, Delta_x, Delta_y, Delta_z, W_sat, m, n', real=True)
w_ijk, w_ijk1, w_ij1k, w_i1jk = smp.symbols('w_{ijk}, w_{ij(k+1)}, w_{i(j+1)k}, w_{(i+1)jk} ', real=True)
w_ijk_1, w_ij_1k, w_i_1jk = smp.symbols('w_{ij(k-1)}, w_{i(j-1)k}, w_{(i-1)jk} ', real=True)

# In[5]:


k_w = K_phi * ((w_ijk - W_sat) / (m - W_sat)) ** (3.5)
k_w

# In[6]:


smp.diff(k_w, w_ijk)

# In[7]:


k_wdiff = 3.5 * K_phi * ((w_ijk - W_sat) / (m - W_sat)) ** (2.5) / (m - W_sat)
k_wdiff

# In[8]:


U = mu * h_k * (-smp.log(abs((w - W_sat) / (m - W_sat)))) ** (1 / n) + z
U

# In[9]:


# Define Drivertives:
dudx = (U.subs(w, w_i1jk) - U.subs(w, w_i_1jk)) / (2 * dx)
dudy = (U.subs(w, w_ij1k) - U.subs(w, w_ij_1k)) / (2 * dy)
dudz = (U.subs([(w, w_ijk1), (z, -dz)]) - U.subs([(w, w_ijk_1), (z, dz)])) / (2 * dz)  # +1

dwdx = (w_i1jk - w_i_1jk) / (2 * dx)
dwdy = (w_ij1k - w_ij_1k) / (2 * dy)
dwdz = (w_ijk1 - w_ijk_1) / (2 * dz)

d2udx2 = (U.subs(w, w_i1jk) + U.subs(w, w_i_1jk) - 2 * U.subs(w, w_ijk)) / (dx ** 2)
d2udy2 = (U.subs(w, w_ij1k) + U.subs(w, w_ij_1k) - 2 * U.subs(w, w_ijk)) / (dy ** 2)
d2udz2 = (U.subs(w, w_ijk1) + U.subs(w, w_ijk_1) - 2 * U.subs(w, w_ijk)) / (dz ** 2)

dudz

# In[10]:


# Differenzial equation
dwdt = +k_wdiff * (dudx * dwdx + dudy * dwdy + dudz * dwdz) - k_w * (d2udx2 + d2udy2 + d2udz2)
dwdt

# In[11]:


# Numerical Values

d_x = 0.2
d_y = 0.2
d_z = 0.2
k_phi = 0.000075
w_sat = 0.17
H_K = 1.25
M = 0.31
MU = 1
N = 3.5

# In[12]:


Usubs = U.subs(
    [(W_sat, w_sat), (h_k, H_K), (dx, d_x), (dy, d_y), (dz, d_z), (m, M), (mu, MU), (K_phi, k_phi), (n, N), (z, 0)])
Usubs

# In[13]:


dwdt_subs = dwdt.subs(
    [(W_sat, w_sat), (h_k, H_K), (dx, d_x), (dy, d_y), (dz, d_z), (m, M), (mu, MU), (K_phi, k_phi), (n, N)])
dwdt_subs

# In[14]:


w_ijk_funct = smp.lambdify([
    w_i1jk,
    w_i_1jk,
    w_ij1k,
    w_ij_1k,
    w_ijk1,
    w_ijk_1,
    w_ijk], dwdt_subs)
w_ijk_funct

# In[15]:


# dimensionen unseres Quaders in x,y,z
n = [14, 14, 5]

# big loop:
delta_t = 1e1
steps = 10000
Zeit = int(steps * delta_t)

# Init
F = np.zeros((n[0], n[1], n[2], steps))
w_initial = 0.19
F.fill(w_initial)

# Start
# w_max = M - 0.01
# F[10:15,10:15,0,-1] = w_max
# F[n[0]-15:n[0]-10,10:15,0,0] = w_max
# F[10:15,n[0]-15:n[0]-10,0,0] = w_max
# F[n[0]-15:n[0]-10,n[0]-15:n[0]-10,0,-1] = w_max

# Start
w_max = 0.29
F[2:3, 2:3, 0, 0] = w_max
F[n[0] - 3:n[0] - 2, 2:3, 0, 0] = w_max
F[2:3, n[0] - 3:n[0] - 2, 0, 0] = w_max
F[n[0] - 3:n[0] - 2, n[0] - 3:n[0] - 2, 0, 0] = w_max


# In[16]:


def w_(x, y, z, t):
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if z < 0:
        z = 0

    if x >= n[0]:
        x = n[0] - 1
    if y >= n[1]:
        y = n[1] - 1
    if z >= n[2]:
        z = n[2] - 1

    return F[x, y, z, t]


# mean_x = n[0]/2
# mean_y = n[1]/2
# count_irrigations = int(round(n[0]*n[1]/9))
# var_x = 10
# var_y = 10
# x_coord = np.round(np.random.randn(count_irrigations)*var_x+mean_x)
# y_coord = np.round(np.random.randn(count_irrigations)*var_y+mean_y)
#
# coord = np.column_stack((x_coord, y_coord))
# coord
#
# coord_list = coord.tolist()
# coord_list
#
# def gaussian_irrigate(t):
#     for element in coord_list:
#         F[int(element[0]),int(element[1]),0,t] = w_max
#
#

# def circle_irrigate(x,y,z,t):
#     value = 0
#     if t<0.1*steps:
#         r = 5
#         m1 = [round(n[0]/4),round((n[1]/4))]
#         m2 = [round(n[0]/4),round((3*n[1]/4))]
#         m3 = [round(3*n[0]/4),round((n[1]/4))]
#         m4 = [round(3*n[0]/4),round(3*(n[1]/4))]
#
#         if (((x-m1[0])**2+(y-m1[1])**2 <= r**2) or ((x-m2[0])**2+(y-m2[1])**2 <= r**2) or ((x-m3[0])**2+(y-m3[1])**2 <= r**2) or ((x-m4[0])**2+(y-m4[1])**2 <= r**2)):
#             value = 0.1
#
#     return value
#
# def irrigate(x,y,z,t):
#     value= 0
#     #if t<(steps/10):
#     if (((x<0.2*n[0] and x>0.1*n[0])or(x<0.9*n[0] and x>0.8*n[0]))and((y<0.2*n[1] and y>0.1*n[1])or(y<0.9*n[1] and y>0.8*n[1]))):
#         value = 0.001
#     return value

# In[17]:


# F[2,2,0] = 0.29


# In[18]:


e = time.time()
print("vorbereitung")
print(e - a)

# In[19]:


anfang = time.time()

irrigation_time = round(steps * 0.2)

# gaussian_irrigate(0)

for step in range(irrigation_time):
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                F[i, j, k, step + 1] = F[i, j, k, step] + delta_t * (w_ijk_funct(w_(i + 1, j, k, step),
                                                                                 w_(i - 1, j, k, step),
                                                                                 w_(i, j + 1, k, step),
                                                                                 w_(i, j - 1, k, step),
                                                                                 w_(i, j, k + 1, step),
                                                                                 w_(i, j, k - 1, step),
                                                                                 w_(i, j, k, step)))

    # F[3,3,0,step] = w_max
    # F[n[0]-3,3,0,step] = w_max
    # F[3,n[0]-3,0,step] = w_max
    # F[n[0]-3,n[0]-3,0,step] = w_max

    if ((step / steps) * 100) % 2 == 0:
        print("{}% calculated".format(step / steps * 100))

print("irrigation done")

for step in range(irrigation_time, steps - 1):
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                F[i, j, k, step + 1] = F[i, j, k, step] + delta_t * w_ijk_funct(w_(i + 1, j, k, step),
                                                                                w_(i - 1, j, k, step),
                                                                                w_(i, j + 1, k, step),
                                                                                w_(i, j - 1, k, step),
                                                                                w_(i, j, k + 1, step),
                                                                                w_(i, j, k - 1, step),
                                                                                w_(i, j, k, step))
    if ((step / steps) * 100) % 2 == 0:
        print("{}% calculated".format(step / steps * 100))

print("Calculation done!")
ende = time.time()

print("Es hat {} Minuten gedauert.".format(ende - anfang))

# In[20]:


# np.log(F[:,:,0,0])
190 / 60

# In[21]:


for t in range(0, steps, int(steps / 10)):
    cmap = plt.cm.get_cmap("Blues").copy()
    cmap.set_bad(color='red')
    plt.imshow(F[:, :, 0, t], cmap=cmap)
    plt.title("Watering at {}s".format(t * delta_t))
    plt.colorbar()
    plt.clim(0.19, 0.29)
    plt.show()

# In[22]:


for t in range(0, steps, 1000):
    cmap = plt.cm.get_cmap("Blues").copy()
    cmap.set_bad(color='red')
    plt.imshow(np.log(F[2, :, :, t].T), cmap=cmap)  # , norm=matplotlib.colors.LogNorm())
    plt.title("Watering at {}s".format(t * delta_t))
    plt.colorbar()
    plt.clim(np.log(0.19), np.log(0.192))
    plt.show()

# def plotheatmap(F_k, k):
#     # Clear the current plot figure
#     plt.clf()
#
#     plt.title(f"Moisture at t = {int(k*delta_t)}s")
#     plt.xlabel("x")
#     plt.ylabel("y")
#
#     # This is to plot u_k (u at time-step k)
#     plt.pcolormesh(F_k, cmap=plt.cm.Blues, vmin=np.min(0.19), vmax=np.max(0.192))
#     plt.colorbar()
#
#     return plt
#
#
# def animate1(k):
#     plotheatmap(F[:,:,0,k], k)
#
#
# anim1 = FuncAnimation(plt.figure(), animate1, interval=1, frames=round(steps), repeat=False)
# anim1.save("Water_animation_first_layer_presentation_new.gif")
#

# def plotheatmap2(F_k, k):
#     # Clear the current plot figure
#     plt.clf()
#
#     plt.title(f"Moisture at t = {k*delta_t:.3f}s")
#     plt.xlabel("x")
#     plt.ylabel("y")
#
#     # This is to plot u_k (u at time-step k)
#     plt.pcolormesh(F_k, cmap=plt.cm.Blues, vmin=np.min(F[:,:,-1,:]), vmax=np.max(F[:,:,-1,:]))
#     plt.colorbar()
#
#     return plt
#
#
# def animate2(k):
#     plotheatmap2(F[:,:,-1,k], k)
#
#
# anim2 = FuncAnimation(plt.figure(), animate2, interval=1, frames=round(steps/10), repeat=False)
# anim2.save("Water_animation_lowest_layer2.gif")

# def plotheatmap3(F_k, k):
#     # Clear the current plot figure
#     plt.clf()
#
#     plt.title(f"Moisture at t = {k*delta_t:.3f}s")
#     plt.xlabel("x")
#     plt.ylabel("z")
#
#     # This is to plot u_k (u at time-step k)
#     plt.pcolormesh(F_k, cmap=plt.cm.Blues, vmin=0.19, vmax=0.192)
#     plt.colorbar()
#
#     return plt
#
#
# def animate3(k):
#     plotheatmap3(F[2,:,::-1,k].T, k)
#
#
# #anim3 = FuncAnimation(plt.figure(), animate3)#, interval=1, frames=round(steps/10), repeat=False)
# #anim3.save("Water_animation_side_layer_presentation.gif")

# In[23]:


## save the vector to a file:
import time

name = "test_" + str(round(time.time()))

with open(name + ".npy", "wb") as f:
    np.save(f, F)

# In[ ]:


# In[24]:


e = time.time()
print(e - a)
print("sekunden")

# In[ ]:


# In[ ]:


# In[25]:


F[:, :, 0, 0]


# In[26]:


def log_transform(im):
    '''returns log(image) scaled to the interval [0,1]'''
    try:
        (min, max) = (im[im > 0].min(), im.max())
        if (max > min) and (max > 0):
            return (np.log(im.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
    except:
        pass
    return im


# In[42]:


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
from pylab import figure, cm
from matplotlib.colors import LogNorm

fig = plt.figure(figsize=(6, 4))

grid = AxesGrid(fig, 111,
                nrows_ncols=(1, 4),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
t = 0
for ax in grid:
    ax.set_axis_off()
    im = ax.imshow(log_transform(F[:, :, 0, t]), cmap=cmap)  # , norm=LogNorm(vmin=0, vmax=1.1))
    t += int(steps / 4)

# when cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]

cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)
cmap = plt.cm.get_cmap("Blues").copy()
cmap.set_bad(color='red')

cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
cbar.ax.set_yticklabels(['19%', '24%', '29%'])
plt.show()

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:




