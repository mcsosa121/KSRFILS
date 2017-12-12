import os
import time
import multiprocessing
import numpy
from krypy.linsys import LinearSystem, Cg
from krypy.deflation import DeflatedCg, DeflatedGmres, Ritz
from krypy.utils import Arnoldi, ritz, BoundCG, ConvergenceError, ArgumentError
from krypy.recycling import RecyclingCg
from krypy.recycling.factories import RitzFactory,RitzFactorySimple
from krypy.recycling.evaluators import RitzApriori,RitzApproxKrylov
from scipy import random, linalg,io,sparse
import matplotlib.pyplot as plt
import pickle

def find_deflation_subspace(A,b,k,ortho='dmgs',ritz_type='ritz'):
    Ar = Arnoldi(A,b,ortho=ortho)
    for i in range(1,k+1):
        Ar.advance()
    [V,H] = Ar.get()
    [theta,U,resnorm,Z] = ritz(H,V,type=ritz_type)
    return Z

def reuse_deflation_subspace(sol,ritz_type='ritz'):
    [theta,U,resnorm,Z] = ritz(sol.H,sol.V,type=ritz_type)
    return Z

def experiment(A,b=None,k=10,numSystems=5,rank=1,maxiter=1000):
    #form systems
    matrixSize = A.shape[0]
    if b == None:
        b=numpy.ones((matrixSize, 1))

    Asys = [A]
    for i in range(1,numSystems):
        u = sparse.random(matrixSize, rank, density=0.001)
        Asys.append(Asys[i-1] + numpy.dot(u,u.T))

    systems = []
    for i in range(0,len(Asys)):
        systems.append(LinearSystem(A=Asys[i],b=b,self_adjoint=True,positive_definite=True))

    cg_sol = []
    ts = time.time()
    for i in range(0,len(Asys)):
        cg_sol = [Cg(systems[i],maxiter=maxiter)]
    te = time.time()
    cg_time = (te-ts)*1000
    
    deflated_sol = []
    ts = time.time()
    for i in range(0,len(Asys)):
        U=sparse.linalg.eigsh(Asys[i],k=k)
        deflated_sol = [DeflatedCg(systems[i],U=U[1],maxiter=maxiter)]
    te = time.time()
    deflated_time = (te-ts)*1000

    recycled_sol = []
    vector_factory = RitzFactorySimple(n_vectors=k, which='sm')
    ts = time.time()
    recycler = RecyclingCg(vector_factory=vector_factory)
    for i in range(0,len(Asys)):
        recycled_sol = [recycler.solve(systems[i],maxiter=maxiter)]
    te = time.time()
    recycled_time = (te-ts)*1000

    return [cg_sol,deflated_sol,recycled_sol,cg_time,deflated_time,recycled_time]

cond = 10000000
while(cond>1e5):
    D = (random.rand(1,10000))[0]
    cond = max(D)/min(D)

print(cond)
sizes = [10,25,50,75,100,250,500,750,1000,2500,5000,7500,10000]
D.sort()
for size in sizes:
    print(size)
    l = numpy.linspace(0,9999,num=size,dtype=int)
    A = numpy.diag(D[l])
    numSystems=10
    [cg_sol,deflated_sol,recycled_sol,cg_time,deflated_time,recycled_time] = experiment(A,k=3,numSystems=numSystems,rank=1,maxiter=1000)
    itemlist = [cg_sol[0].resnorms,deflated_sol[0].resnorms,recycled_sol[0].resnorms,cg_time,deflated_time,recycled_time,A.shape[0],cond]
    with open('./size/'+str(size)+'.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)
