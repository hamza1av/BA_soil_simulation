#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as smp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import time
import random
#from IPython.display import display, HTML

#display(HTML("<style>.container { width:99% !important; }</style>"))
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [40, 20]
from numba import jit, njit

# # Formel vom Paper
# $\frac{\delta}{\delta t} W= k'(W) \Big( \frac{\delta}{\delta x} U \frac{\delta}{\delta x} W + \frac{\delta}{\delta y} U \frac{\delta}{\delta y} W + \frac{\delta}{\delta z}U \frac{\delta}{\delta z} W \Big) + k(W)\Big(\frac{\delta^2}{\delta x^2}U + \frac{\delta^2}{\delta y^2}U + \frac{\delta^2}{\delta z^2}U \Big) + f(x,y,z,t)$
#
# # Änderung:
# $\frac{\delta}{\delta t} W= k'(W) \Big( \frac{\delta}{\delta x} U \frac{\delta}{\delta x} W + \frac{\delta}{\delta y} U \frac{\delta}{\delta y} W + \frac{\delta}{\delta z}U \frac{\delta}{\delta z} W \Big) - k(W)\Big(\frac{\delta^2}{\delta x^2}U + \frac{\delta^2}{\delta y^2}U + \frac{\delta^2}{\delta z^2}U \Big) + f(x,y,z,t)$
#
# Probelm: $U(W)$ sinkt mit steigendem W, ein postitives $\frac{\partial^2W}{\partial x ^ 2}$ fürt damit zu einem negativen Wert für  $\frac{\partial^2U}{\partial x ^ 2}$
#
# Jedoch soll bei einem n postitiven $\frac{\partial^2W}{\partial x ^ 2}$ der wert für $W$ steigen, also muss $\frac{\delta}{\delta t} W>0$ sein. Damit feht dort irgendwo ein Minus

# In[2]:


t, mu, h_k, K_phi, x, y, z, W_sat, m, n = smp.symbols('t mu h_k K_phi x y z W_sat m n', real=True)
w, w_ap, dt, mu, h_k, K_phi, dx, dy, dz, W_sat, m, n = smp.symbols(
    'W, W_{AP} Delta_t, mu, h_k, K_phi, Delta_x, Delta_y, Delta_z, W_sat, m, n', real=True)
w_ijk, w_ijk1, w_ij1k, w_i1jk = smp.symbols('w_{ijk}, w_{ij(k+1)}, w_{i(j+1)k}, w_{(i+1)jk} ', real=True)
w_ijk_1, w_ij_1k, w_i_1jk = smp.symbols('w_{ij(k-1)}, w_{i(j-1)k}, w_{(i-1)jk} ', real=True)

# In[3]:


k_w = K_phi * ((w_ijk - W_sat) / (m - W_sat)) ** (3.5)
k_w

# In[4]:


smp.diff(k_w, w_ijk)

# In[5]:


k_wdiff = 3.5 * K_phi * ((w_ijk - W_sat) / (m - W_sat)) ** (2.5) / (m - W_sat)
k_wdiff

# In[6]:


U = mu * h_k * (-smp.log(abs((w - W_sat) / (m - W_sat)))) ** (1 / n) + z
U

# In[ ]:


# In[7]:


# Define Drivertives:
dudx = (U.subs(w, w_i1jk) - U.subs(w, w_i_1jk)) / (2 * dx)
dudy = (U.subs(w, w_ij1k) - U.subs(w, w_ij_1k)) / (2 * dy)
dudz = (U.subs(w, w_ijk1) - U.subs(w, w_ijk_1)) / (2 * dz)  # +1

dwdx = (w_i1jk - w_i_1jk) / (2 * dx)
dwdy = (w_ij1k - w_ij_1k) / (2 * dy)
dwdz = (w_ijk1 - w_ijk_1) / (2 * dz)

d2udx2 = (U.subs(w, w_i1jk) + U.subs(w, w_i_1jk) - 2 * U.subs(w, w_ijk)) / (dx ** 2)
d2udy2 = (U.subs(w, w_ij1k) + U.subs(w, w_ij_1k) - 2 * U.subs(w, w_ijk)) / (dy ** 2)
d2udz2 = (U.subs(w, w_ijk1) + U.subs(w, w_ijk_1) - 2 * U.subs(w, w_ijk)) / (dz ** 2)

#####dudz

# In[8]:


# Differenzial equation
dwdt = k_wdiff * (dudx * dwdx + dudy * dwdy + dudz * dwdz) - k_w * (d2udx2 + d2udy2 + d2udz2)
#####dwdt
# print(smp.python(dwdt))


# In[ ]:


# In[9]:


# Numerical Values

Delta_x = 0.1
Delta_y = 0.1
Delta_z = 0.1
K_phi = 0.000075
w_sat = 0.17
H_K = 1.25
M = 0.31
MU = 1
N = 3.5

# In[10]:


# Usubs = U.subs([(W_sat,w_sat), (h_k,H_K), (dx , d_x), (dy , d_y), (dz ,d_z), (m,M), (mu,MU), (K_phi,K_phi), (n,N),(z,0)])
# Usubs


# In[11]:


# U_func = smp.lambdify(w,Usubs)
# w_num = np.linspace(w_sat,M,100)
# u_num = np.zeros(100)
# for i in range(len(w_num)):
#    u_num[i] = U_func(w_num[i])

# plt.plot(w_num,u_num)


# In[12]:


dwdt_subs = dwdt.subs(
    [(W_sat, w_sat), (h_k, H_K), (dx, Delta_x), (dy, Delta_y), (dz, Delta_z), (m, M), (mu, MU), (K_phi, K_phi), (n, N)])
#####dwdt_subs

# In[13]:


# WAP = 0.2
# smp.diff(dwdt_subs,w_ijk).subs({w_ijk: WAP, w_i1jk: WAP, w_ij1k: WAP, w_ijk1: WAP, w_i_1jk:WAP, w_ij_1k:WAP, w_ijk_1:WAP})


# In[14]:


w_ijk_funct = smp.lambdify([
    w_i1jk,
    w_i_1jk,
    w_ij1k,
    w_ij_1k,
    w_ijk1,
    w_ijk_1,
    w_ijk], dwdt_subs)
####w_ijk_funct

# In[15]:


# dimensionen unseres Quaders in x,y,z
n = [10, 20, 5]

# big loop:
delta_t = 0.01
steps = 500
Zeit = int(steps * delta_t)

# Init
F = np.zeros((n[0], n[1], n[2], steps))
w_initial = (w_sat + M) / 2
F.fill(w_initial)

# Start
w_max = m - 0.01


# F[2:3,2:3,0,0] = w_max
# F[n[0]-3:n[0]-2,2:3,0,0] = w_max
# F[2:3,n[0]-3:n[0]-2,0,0] = w_max
# F[n[0]-3:n[0]-2,n[0]-3:n[0]-2,0,0] = w_max

# for i in range(50):
#   F[random.randint(0,n[0]-1),random.randint(0,n[1]-1),0,0] = w_max


# In[16]:


# @njit
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


def irrigate(x, y, z, t):
    value = 0
    if t < (steps / 10):
        if (((x < 0.2 * n[0] and x > 0.1 * n[0]) or (x < 0.9 * n[0] and x > 0.8 * n[0])) and (
                (y < 0.2 * n[1] and y > 0.1 * n[1]) or (y < 0.9 * n[1] and y > 0.8 * n[1]))):
            value = 1
    return value


# @njit
def circle_irrigate(x, y, z, t):
    value = 0
    if t < 0.1 * steps:
        r = 5
        m1 = [round(n[0] / 4), round((n[1] / 4))]
        m2 = [round(n[0] / 4), round((3 * n[1] / 4))]
        m3 = [round(3 * n[0] / 4), round((n[1] / 4))]
        m4 = [round(3 * n[0] / 4), round(3 * (n[1] / 4))]

        if (((x - m1[0]) ** 2 + (y - m1[1]) ** 2 <= r ** 2) or ((x - m2[0]) ** 2 + (y - m2[1]) ** 2 <= r ** 2) or (
                (x - m3[0]) ** 2 + (y - m3[1]) ** 2 <= r ** 2) or ((x - m4[0]) ** 2 + (y - m4[1]) ** 2 <= r ** 2)):
            value = 0.1

    return value


# In[17]:


amount = 10
r = 1


def random_circles(x, y, z, t):
    if t < 0.1 * steps:
        circles_buff = np.array([np.random.randint(n[0], size=amount), np.random.randint(n[1], size=amount)]).T
        circles = circles_buff.tolist()
        for m in circles:
            if ((x - m[0]) ** 2 + (y - m[1]) ** 2 <= r ** 2):
                return 0.1


# In[18]:


import time

anfang = time.time()

for step in range(steps - 1):
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                F[i, j, k, step + 1] = F[i, j, k, step] + delta_t * (
                            circle_irrigate(i, j, k, step) + w_ijk_funct(w_(i + 1, j, k, step),
                                                                         w_(i - 1, j, k, step),
                                                                         w_(i, j + 1, k, step),
                                                                         w_(i, j - 1, k, step),
                                                                         w_(i, j, k + 1, step),
                                                                         w_(i, j, k - 1, step),
                                                                         w_(i, j, k, step)))
    # if step<100:
    #   F[round(0.1*n[0]):round(0.15*n[0]),round(0.1*n[1]):round(0.15*n[1]),0,step] = 0.3
    #  F[round(0.85*n[0]):round(0.9*n[0]),round(0.1*n[1]):round(0.15*n[1]),0,step] = 0.3
    # F[round(0.1*n[0]):round(0.15*n[0]),round(0.85*n[1]):round(0.9*n[1]),0,step] = 0.3
    # F[round(0.85*n[0]):round(0.9*n[0]),round(0.85*n[1]):round(0.9*n[1]),0,step] = 0.3

    # for i in range(100):
    # F[random.randint(0,n[0]-1),random.randint(0,n[1]-1),0,step] = w_max

    if ((step / steps) * 100) % 1 == 0:
        print("{}% calculated".format(step / steps * 100))
print("100% calculated")
zeit = round(time.time() - anfang)
print("It took {}s".format(zeit))

# In[ ]:


for t in range(0, steps, round(steps / 20)):
    z = 0
    # t = 1
    cmap = plt.cm.get_cmap("Blues").copy()
    cmap.set_bad(color='red')
    plt.imshow(F[:, 10, :, t].T, cmap=cmap)
    plt.colorbar()
    # plt.clim(0.0,0.50)
    plt.show()

cmap = plt.cm.get_cmap("Blues").copy()
cmap.set_bad(color='red')
plt.imshow(F[:, :, z, -1], cmap=cmap)
plt.colorbar()
plt.show()

# In[ ]:


from IPython.core.display import display, HTML

display(HTML("<style>div.output_scroll { height: 66em; }</style>"))

# In[ ]:


# In[ ]:


# In[ ]:


for t in range(0, steps, round(steps / 20)):
    z = 0
    # t = 1
    cmap = plt.cm.get_cmap("Blues").copy()
    cmap.set_bad(color='red')
    plt.imshow(F[:, :, -0, t], cmap=cmap)
    plt.colorbar()
    plt.show()

cmap = plt.cm.get_cmap("Blues").copy()
cmap.set_bad(color='red')
plt.imshow(F[:, :, z, -1], cmap=cmap)
plt.colorbar()
plt.show()

# In[ ]:


F

# In[ ]:


# In[ ]:


# with open("BA_Simulation_100_100_5_4000_random_rain.json","w") as file:
#    json.dump(F.tolist(),file)


# In[ ]:


# In[ ]:


cmap = plt.cm.get_cmap("Blues").copy()
cmap.set_bad(color='red')
plt.imshow(F[:, :, 0, 100], cmap=cmap)
plt.colorbar()
plt.show()

# In[ ]:


fig = plt.figure()
ax = plt.axes()


def animate(i):
    return F[:, :, 0, i]


anim = animation.FuncAnimation(fig, animate,
                               frames=100, interval=20, blit=True)

# In[ ]:




