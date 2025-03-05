import numpy as np

F = np.load("F_19_19_7_1000.npy")

for i in range(19):
    for j in range(19):
        #print("({},{},{})".format(i*0.2,j*0.2,F[i,j,0,0]))
        for i_2 in range(10):
            for j_2 in range(10):
                print("({},{},{})".format(i*0.2+0.02*i_2,j*0.2+0.02*j_2,F[i,j,0,0]))