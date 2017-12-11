import os
import numpy
import matplotlib.pyplot as plt
import pickle
from scipy import linalg,io,sparse

dir_name = './results2/'
directory = os.fsencode(dir_name)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
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
    tol = 0
    while True:
        try:
            eig1 = sparse.linalg.eigs(A,k=1,which='LM',return_eigenvectors=False,maxiter=20000,tol=tol)
            break
        except sparse.linalg.ArpackNoConvergence:
            print('eig1 did not converge, try again')
            if tol==0:
                tol=1e-13
            else:
                tol = tol*100

    tol = 0
    while True:
        try:
            eig2 = sparse.linalg.eigs(A,k=1,which='SM',return_eigenvectors=False,maxiter=20000,tol=tol)
            break
        except sparse.linalg.ArpackNoConvergence:
            print('eig2 did not converge, try again')
            if tol==0:
                tol=1e-13
            else:
                tol = tol*100
    cond = abs(eig1/eig2)
    size = A.shape[0]
    sclist = [size,cond]
    with open('./res_sc/'+filename,'wb') as fp:
        pickle.dump(sclist, fp)
