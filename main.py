import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from numba import jit
anfang = time.time()
### establishing the dimensions of the field
lenX = 100
lenY = 100
field = np.empty((lenX,lenY))
field.fill(0.0)

delta_x = 1
delta_y = 1
maxIter = 2_000
delta_t = 1

K =  1.5

X,Y = np.meshgrid(np.arange(0,lenX),np.arange(0,lenY))

field[23:27,23:27]= 100
field[23:27,73:77]= 100
field[73:77,23:27]= 100
field[73:77,73:77]= 100

F = field

clr_interp = 50
cMap = plt.cm.Blues


### i = x values
### j = y values


fig = plt.figure()
Bild = plt.contourf(X,Y,F,clr_interp,cmap= cMap)

@jit(nopython=True)
def update_field(F, lenX, lenY, delta_x, delta_y):
    for i in range(1, lenX - 1):
        for j in range(1, lenY - 1):
            F[i,j] = ((delta_y**2*(F[i+1,j]+F[i-1,j]) + delta_x**2*(F[i,j+1]+F[i,j-1]))/(2*(delta_y+delta_x)))
    return F

for iteration in range(0,maxIter):
    F = update_field(F, lenX, lenY, delta_x, delta_y)
    if np.average(F) < 18:
        F[23:27, 23:27] = 100
        F[23:27, 73:77] = 100
        F[73:77, 23:27] = 100
        F[73:77, 73:77] = 100

print(np.mean(F))
ende = round(time.time()-anfang,1)
print(f"It took {ende}s")

plt.title("2D plot Bodenfeuchtigkeit")
plt.contourf(X,Y,F,clr_interp,cmap= cMap)
plt.colorbar()
plt.show()
