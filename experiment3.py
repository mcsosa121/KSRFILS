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
        u = random.rand(matrixSize, rank)
        Asys.append(Asys[i-1] + numpy.dot(u,u.T))

    systems = []
    for i in range(0,len(Asys)):
        systems.append(LinearSystem(A=Asys[i],b=b,self_adjoint=True,positive_definite=True))

    cg_sol = []
    ts = time.time()
    for i in range(0,len(Asys)):
        cg_sol.append(Cg(systems[i],maxiter=maxiter))
    te = time.time()
    cg_time = (te-ts)*1000

    deflated_sol = []
    ts = time.time()
    for i in range(0,len(Asys)):
        U=find_deflation_subspace(Asys[i],b,k)
        deflated_sol.append(DeflatedCg(systems[i],U=U,maxiter=maxiter))
    te = time.time()
    deflated_time = (te-ts)*1000

    recycled_sol = []
    vector_factory = RitzFactorySimple(n_vectors=k, which='sm')
    ts = time.time()
    recycler = RecyclingCg(vector_factory=vector_factory)
    for i in range(0,len(Asys)):
        recycled_sol.append(recycler.solve(systems[i],maxiter=maxiter))
    te = time.time()
    recycled_time = (te-ts)*1000

    return [cg_sol,deflated_sol,recycled_sol,cg_time,deflated_time,recycled_time]

matrixSize = 100
R = random.rand(matrixSize,matrixSize)
A = numpy.dot(R,R.transpose())

[u,s,v] = numpy.linalg.svd(A)
print(numpy.linalg.cond(u*numpy.diag(s)*v))
for i in range(0,10):
    print(i)
    s[0]=s[0]*5
    A = u*numpy.diag(s)*v
    numSystems=10
    [cg_sol,deflated_sol,recycled_sol,cg_time,deflated_time,recycled_time] = experiment(A,numSystems=numSystems,k=2,maxiter=10000)
    cond = numpy.linalg.cond(A)
    itemlist = [cg_sol[numSystems-1].resnorms,deflated_sol[numSystems-1].resnorms,recycled_sol[numSystems-1].resnorms,cg_time,deflated_time,recycled_time,A.shape[0],cond]
    with open('./cond/'+str(cond)+'.txt', 'wb') as fp:
        pickle.dump(itemlist, fp)
