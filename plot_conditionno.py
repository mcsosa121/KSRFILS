import os
import numpy
import matplotlib.pyplot as plt
import pickle
from scipy import linalg,io,sparse


def plot_with_trend(x,y,marker,label,color,deg):
    plt.plot(x,y,marker=marker,label=label,color=color,linestyle='None')
    z = numpy.polyfit(x, y, deg=deg)
    p = numpy.poly1d(z)
    plt.plot(x,p(x),"--")

size_list = []
cond_list = []
cg_tlist = []
deflated_tlist = []
recycled_tlist = []
cg_llist = []
deflated_llist = []
recycled_llist = []

dir_name = './results2/'
directory = os.fsencode(dir_name)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    #print(filename)
    with open(dir_name+filename, 'rb') as fp:
        itemlist = pickle.load(fp)
    name = filename[4:]
    Mat = io.loadmat('./done/'+name[:-2]+'.mat')
    A = None
    for i in range(0,len(Mat['Problem'][0][0])):
        if type(Mat['Problem'][0][0][i]) == sparse.csc.csc_matrix:
            A = Mat['Problem'][0][0][i]
            break
    if A==None:
        print('Could not find matrix of',filename)
        continue
    with open('./res_sc/'+filename,'rb') as fp:
        sclist = pickle.load(fp)
    size_list.append(sclist[0])
    cond_list.append(sclist[1][0])
    cg_tlist.append(itemlist[-3])
    deflated_tlist.append(itemlist[-2])
    recycled_tlist.append(itemlist[-1])
    cg_llist.append(len(itemlist[-6]))
    deflated_llist.append(len(itemlist[-5]))
    recycled_llist.append(len(itemlist[-4]))

size_list = numpy.array(size_list)
cond_list = numpy.array(cond_list)
cg_tlist = numpy.array(cg_tlist)
deflated_tlist = numpy.array(deflated_tlist)
recycled_tlist = numpy.array(recycled_tlist)
cg_llist = numpy.array(cg_llist)
deflated_llist = numpy.array(deflated_llist)
recycled_llist = numpy.array(recycled_llist)
inds_sizes = size_list.argsort()
inds_cond = cond_list.argsort()

plt.title('Size of matrix against time taken')
plot_with_trend(size_list[inds_sizes],cg_tlist[inds_sizes],marker='x',label='cg',color='blue',deg=1)
plot_with_trend(size_list[inds_sizes],deflated_tlist[inds_sizes],marker='^',label='deflated',color='orange',deg=1)
plot_with_trend(size_list[inds_sizes],recycled_tlist[inds_sizes],marker='o',label='recycled',color='green',deg=1)
plt.yscale('log')
plt.ylabel('Time (ms)')
plt.xscale('log')
plt.xlabel('Size of matrix')
plt.legend()
plt.show()

plt.title('Size of matrix against number of iterations')
plot_with_trend(size_list[inds_sizes],cg_llist[inds_sizes],marker='x',label='cg',color='blue',deg=1)
plot_with_trend(size_list[inds_sizes],deflated_llist[inds_sizes],marker='^',label='deflated',color='orange',deg=1)
plot_with_trend(size_list[inds_sizes],recycled_llist[inds_sizes],marker='o',label='recycled',color='green',deg=1)
plt.yscale('log')
plt.ylabel('Number of iterations')
plt.xscale('log')
plt.xlabel('Size of matrix')
plt.legend()
plt.show()

plt.title('Condition number of matrix against time taken')
plot_with_trend(cond_list[inds_cond],cg_tlist[inds_cond],marker='x',label='cg',color='blue',deg=1)
plot_with_trend(cond_list[inds_cond],deflated_tlist[inds_cond],marker='^',label='deflated',color='orange',deg=1)
plot_with_trend(cond_list[inds_cond],recycled_tlist[inds_cond],marker='o',label='recycled',color='green',deg=1)
plt.yscale('log')
plt.ylabel('Time (ms)')
plt.xscale('log')
plt.xlabel('Condition number of matrix')
plt.legend()
plt.show()

plt.title('Condition number against number of iterations')
plot_with_trend(cond_list[inds_cond],cg_llist[inds_cond],marker='x',label='cg',color='blue',deg=1)
plot_with_trend(cond_list[inds_cond],deflated_llist[inds_cond],marker='^',label='deflated',color='orange',deg=1)
plot_with_trend(cond_list[inds_cond],recycled_llist[inds_cond],marker='o',label='recycled',color='green',deg=1)
plt.yscale('log')
plt.ylabel('Number of iterations')
plt.xscale('log')
plt.xlabel('Condition number of matrix')
plt.legend()
plt.show()
