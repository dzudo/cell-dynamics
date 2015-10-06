import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from heapq import nlargest
import math
import Image

def get_labels(filename):
    print filename
    seg = pickle.load(open(filename, 'r'))

    (sx, sy) = seg.shape

    labels = range(0,sx*sy)

    label = np.zeros((sx,sy))

    def find(x):
        while labels[x] != x:
            x = labels[x]
        return x

    def union(x, y):
        labels[find(x)] = find(y)

    largest_label = 0
    for x in range(0, sx): 
        for y in range(0, sy):
            if seg[x,y] >= 0: 
                left = 1 if seg[x-1,y] >= 0 else 0
                above = 1 if seg[x,y-1] >= 0 else 0
                if (left == 0) and (above == 0):
                    largest_label = largest_label + 1
                    label[x,y] = largest_label
                else:
                    if (left != 0):
                        if (above != 0):
                            union(max(int(label[x-1,y]),int(label[x,y-1])), min(int(label[x-1,y]), int(label[x,y-1])));
                        label[x,y] =  find(int(label[x-1,y]))
                    else:
                        label[x,y] = find(int(label[x,y-1]))


    new_labels=[]
    counts=[]

    for x in range(sx):
        for y in range(sy):
            label[x,y]=find(int(label[x,y]))
            if int(label[x,y]) not in new_labels:
                new_labels.append(int(label[x,y]))
                counts.append(0)
            label[x,y] = new_labels.index(int(label[x,y]))
            counts[int(label[x,y])]+=1

    return label, counts

con_files_w3 = sorted([os.path.join(root, name) for root, dirs, files in
os.walk("/media/dzudo/sharecenter/DZUDO/mdrab/res_w3_lset/") for
        name in files if name.endswith(".con")])


con_files_w1 = sorted([os.path.join(root, name) for root, dirs, files in
os.walk("/media/dzudo/sharecenter/DZUDO/mdrab/res_w1_lset/") for
        name in files if name.endswith(".con")])


con_files_w2 = sorted([os.path.join(root, name) for root, dirs, files in
os.walk("/media/dzudo/sharecenter/DZUDO/mdrab/res_w2_lset/") for
        name in files if name.endswith(".con")])


con_files_w1_in = sorted([os.path.join(root, name) for root, dirs, files in
os.walk("/media/dzudo/sharecenter/DZUDO/mdrab/res_w1_lset_in/") for
        name in files if name.endswith(".con")])

bacterial_centres = []
bacterial_vals = []
nuclei_centres = []
cavioli_centres = []
infection_centres = []
infection_vals = []

im_index = 0

for filename in con_files_w3:
    print filename
    labeled_segments, counts = get_labels(filename)
    (sx, sy) = labeled_segments.shape

    filename_w1 = filename.replace('w3','w1')
    cell_shape = pickle.load(open(filename_w1, 'r'))

    max_num = nlargest(3, counts)[1:]
    max_label = nlargest(3, range(len(counts)), key = lambda k : counts[k])[1:]
    print "MAX LABELS: ", max_label
    print "MAX NUMS: ", max_num

    values = [[],[]]
    values_inf = [[],[]]
    centres = []
    centres_inf = []
    sizes_inf = []
    if im_index >= 24:
        for x in range(sx):
            for y in range(sy):
                if int(labeled_segments[x,y]) in max_label:
                    values[max_label.index(int(labeled_segments[x,y]))].append([x,y])
                    if cell_shape[x,y] >=0:
                        values_inf[max_label.index(int(labeled_segments[x,y]))].append([x,y])

        for v_list in values:
            centres.append([sum([x[0] for x in v_list])/float(len(v_list)),sum([x[1]
                for x in v_list])/float(len(v_list))])

    
        for v_list in values_inf:
            if len(v_list) > 0:
                centres_inf.append([sum([x[0] for x in v_list])/float(len(v_list)),sum([x[1]
                for x in v_list])/float(len(v_list))])
            else:
                centres_inf.append([0,0])
            sizes_inf.append(len(v_list))
    else:
        centres = [[0,0],[0,0]]
        centres_inf = [[0,0],[0,0]]
        max_num = [0,0]
        sizes_inf = [0,0]

    im_index += 1


    print "CENTRES: ", centres

    Image.open("/media/dzudo/sharecenter/DZUDO/mdrab/dane_w1/png/dane_w1" +
            filename.split("/")[-1].rstrip("_seg.png.con").lstrip("dane_w3_") +
    ".png").resize((500,500), Image.ANTIALIAS).save("tmp.png")


    plt.imshow(plt.imread("tmp.png"), cmap='gray')
    plt.draw()
    plt.hold(True)
    contour = pickle.load(open(filename, 'r'))
    plt.contour(contour, 0, colors='r')
    plt.contour(cell_shape, 0, colors='b')
    plt.draw()
    for i, centre in enumerate(centres):
        circle1 = plt.Circle((centre[1],centre[0]), math.sqrt(max_num[i]/math.pi),
                ec = 'g', fc = 'none')
        plt.gca().add_artist(circle1)

    
    for i, centre in enumerate(centres_inf):
        circle2 = plt.Circle((centre[1],centre[0]), math.sqrt(sizes_inf[i]/math.pi),
                ec = 'y', fc = 'none')
        plt.gca().add_artist(circle2)

    plt.savefig("vec_png/" + filename.split("/")[-1].strip(".con"))
    plt.hold(False)

    bacterial_centres.append([[centre[0]*1./sx, centre[1]*1./sy]for centre in centres])
    bacterial_vals.append([num *1./(sx*sy) for num in max_num])

    
    infection_centres.append([[centre[0]*1./sx, centre[1]*1./sy]for centre in
        centres_inf])
    infection_vals.append([num *1./(sx*sy) for num in sizes_inf])

for confile in con_files_w1_in:
    seg = pickle.load(open(confile, "r"))
    vals = []
    (sx,sy) = seg.shape
    for x in range(sx):
        for y in range(sy):
            if seg[x,y] >= 0:
                vals.append([x,y])
    centre = [sum([val[0]*1./sx for val in vals])/len(vals), sum([val[1]*1./sy for val in vals])/len(vals)]
    print centre
    nuclei_centres.append(centre)


for confile in con_files_w2:
    seg = pickle.load(open(confile, "r"))
    vals = []
    (sx,sy) = seg.shape
    for x in range(sx):
        for y in range(sy):
            if seg[x,y] >= 0:
                vals.append([x,y])
    centre = [sum([val[0]*1./sx for val in vals])/len(vals), sum([val[1]*1./sy for val in vals])/len(vals)]
    print centre
    cavioli_centres.append(centre)

U1 = []
for i in range(len(bacterial_centres)):
    u = []
    for j in range(len(bacterial_centres[i])):
        u.append(nuclei_centres[i][0] - bacterial_centres[i][j][0])
        u.append(nuclei_centres[i][1] - bacterial_centres[i][j][1])
        u.append(bacterial_vals[i][j])
    U1.append(np.matrix(u).T)


U2 = []
for i in range(len(infection_centres)):
    u = []
    for j in range(len(infection_centres[i])):
        u.append(nuclei_centres[i][0] - infection_centres[i][j][0])
        u.append(nuclei_centres[i][1] - infection_centres[i][j][1])
        u.append(infection_vals[i][j])
    U2.append(np.matrix(u).T)

Y = [np.matrix([nuclei_centres[i][0] - cavioli_centres[i][0], nuclei_centres[i][1] -
    cavioli_centres[i][1]]).T for i in range(len(nuclei_centres))]

pickle.dump(U1, open('U1', 'w+'))
pickle.dump(U2, open('U2', 'w+'))
pickle.dump(Y, open('UY', 'w+'))




