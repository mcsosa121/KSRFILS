# -*- coding: utf-8 -*-
import time
import numpy
from krypy.linsys import LinearSystem, Cg
from krypy.deflation import DeflatedCg, DeflatedGmres, Ritz
from krypy.utils import Arnoldi, ritz, BoundCG
from krypy.recycling import RecyclingCg
from krypy.recycling.factories import RitzFactory,RitzFactorySimple
from krypy.recycling.evaluators import RitzApriori,RitzApproxKrylov
from scipy import random, linalg

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

cgt = []
dft = []
rct = []
for i in range(1,100):
    matrixSize = 100
    R = random.rand(matrixSize,matrixSize)
    A = numpy.dot(R,R.transpose())
    b=numpy.ones((matrixSize, 1))
    k = 10
    numSystems = 10
    rank = 1 #rank of each system to add
    Asys = [A]
    for i in range(1,numSystems):
        u = random.rand(matrixSize, rank)
        Asys.append(Asys[i-1] + numpy.dot(u,u.T))

    systems = []
    for i in range(0,len(Asys)):
        systems.append(LinearSystem(A=Asys[i],b=b,self_adjoint=True,positive_definite=True))

    ts = time.time()
    for i in range(0,len(Asys)):
        cg_sol = Cg(systems[i],maxiter=1000)
    te = time.time()
    cgt.append((te-ts)*1000)

    ts = time.time()
    for i in range(0,len(Asys)):
        U=find_deflation_subspace(Asys[i],b,k)
        deflated_sol = DeflatedCg(systems[i],U=U,maxiter=1000)
    te = time.time()
    dft.append((te-ts)*1000)

    vector_factory = RitzFactorySimple(n_vectors=k, which='sm')
    ts = time.time()
    recycler = RecyclingCg(vector_factory=vector_factory)
    for i in range(0,len(Asys)):
        recycled_sol =  recycler.solve(systems[i],maxiter=1000)
    te = time.time()
    rct.append((te-ts)*1000)

print('Mean time taken for CG (ms):', sum(cgt)/len(cgt))
print('Mean time taken for Deflated CG (ms):', sum(dft)/len(dft))
print('Mean time taken for Recycled CG (ms):', sum(rct)/len(rct))
