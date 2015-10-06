import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from heapq import nlargest
import math
import Image

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

plt.ion()

#con_files_w3 = sorted([os.path.join(root, name) for root, dirs, files in
#os.walk("/media/dzudo/sharecenter/DZUDO/mdrab/res_w3_lset/") for
#        name in files if name.endswith(".con")])

U1 = pickle.load(open('U1','r'))
U2 = pickle.load(open('U2','r'))
Y = pickle.load(open('UY','r'))

im_index =  0

t=np.arange(len(Y))

plt.subplot(321)

plt.plot(t, [math.atan2(u[1],u[0]) for u in U1], '-g')
plt.plot(t, [math.atan2(u[4],u[3]) for u in U1], '-b')


plt.subplot(322)

plt.plot(t, [abs(u[2][0,0]) for u in U1], '-g')
plt.plot(t, [abs(u[5][0,0]) for u in U1], '-b')


plt.subplot(323)

plt.plot(t, [math.atan2(u[1],u[0]) for u in U2], '-g')
plt.plot(t, [math.atan2(u[4],u[3]) for u in U2], '-b')

plt.subplot(324)

plt.plot(t, [abs(u[2][0,0]) for u in U2], '-g')
plt.plot(t, [abs(u[5][0,0]) for u in U2], '-b')

plt.subplot(325)

plt.plot(t, [math.atan2(u[1],u[0]) for u in Y], '-g')



plt.savefig('data.png')

raw_input()

"""
for filename in con_files_w3:
    print filename
    Image.open("/media/dzudo/sharecenter/DZUDO/mdrab/dane_w1/png/dane_w1" +
            filename.split("/")[-1].rstrip("_seg.png.con").lstrip("dane_w3_") +
    ".png").resize((500,500), Image.ANTIALIAS).save("tmp.png")

    plt.clf()
    ax1 = plt.subplot(221)

    cell_shape = pickle.load(open(filename.replace('w3','w1'),'r'))

    #nuclei_shape = pickle.load(open(filename.replace('w3','w1'),'r'))


    ax1.imshow(plt.imread("tmp.png"), cmap='gray')
    plt.hold(True)
    contour = pickle.load(open(filename, 'r'))
    plt.contour(contour, 0, colors='r')
    plt.contour(cell_shape, 0, colors='b')
    plt.draw()

    ax2 = plt.subplot(222)

    centre = U1[im_index]
    print [centre[0]],[centre[1]]
    plt.plot(np.array(centre[0]),np.array(centre[1]),'or')
    #aplt.gca().add_artist(circle1)

    #circle2 = plt.Circle((centre[3],centre[4]), 2, ec = 'r', color = 'r')
    #plt.gca().add_artist(circle2)

    plt.draw()

    plt.subplot(223)

    centre = U2[im_index]
   
    circle3 = plt.Circle((centre[3],centre[4]), 2, ec = 'g', color = 'g')
    plt.gca().add_artist(circle3)

    circle4 = plt.Circle((centre[3],centre[4]), 2, ec = 'r', color = 'r')
    plt.gca().add_artist(circle4)
    plt.draw()
    
    plt.subplot(224)

    circle5 = plt.Circle((Y[im_index][0,0],Y[im_index][1,0]), 2, ec = 'r', color = 'r')
    plt.gca().add_artist(circle5)


    plt.draw()

    plt.savefig("data_pr/" + filename.split("/")[-1].strip(".con"))
    plt.hold(False)

    im_index += 1
"""



