import os
import numpy
import matplotlib.pyplot as plt
import pickle
from scipy import linalg,io,sparse
from scipy.interpolate import UnivariateSpline

def plot_with_trend(x,y,marker,label,color,deg):
    #trendline is not helpful, removed.
    plt.plot(x,y,marker=marker,label=label,color=color)

size_list = []
cond_list = []
cg_tlist = []
deflated_tlist = []
recycled_tlist = []
cg_llist = []
deflated_llist = []
recycled_llist = []

dir_name = './cond/'
directory = os.fsencode(dir_name)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    with open(dir_name+filename, 'rb') as fp:
        itemlist = pickle.load(fp) 
    cg_llist.append(len(itemlist[0]))
    deflated_llist.append(len(itemlist[1]))
    recycled_llist.append(len(itemlist[2]))
    cg_tlist.append(itemlist[3])
    deflated_tlist.append(itemlist[4])
    recycled_tlist.append(itemlist[5])
    size_list.append(itemlist[6])
    cond_list.append(itemlist[7])

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
plt.ylabel('Number of iterations')
plt.xscale('log')
plt.xlabel('Condition number of matrix')
plt.legend()
plt.show()
