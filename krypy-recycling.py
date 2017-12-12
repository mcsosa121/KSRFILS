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

cg_sol = []
ts = time.time()
for i in range(0,len(Asys)):
    cg_sol.append(Cg(systems[i],maxiter=1000))
te = time.time()
print("CG time taken: ", (te-ts)*1000)

deflated_sol = []
ts = time.time()
for i in range(0,len(Asys)):
    U=find_deflation_subspace(Asys[i],b,k)
    deflated_sol.append(DeflatedCg(systems[i],U=U,maxiter=1000))
te = time.time()
print("Deflated CG (creates multiple subspaces) time taken: ", (te-ts)*1000)

defsingle = []
ts = time.time()
U=find_deflation_subspace(Asys[0],b,k)
defsingle.append(DeflatedCg(systems[0],U=U,maxiter=1000,store_arnoldi=True))
for i in range(1,len(Asys)):
    r = Ritz(defsingle[i-1])
    indices = numpy.argsort(numpy.abs(r.values))[:k]
    U = r.get_vectors(indices)
    defsingle.append(DeflatedCg(systems[i],U=U,maxiter=1000,store_arnoldi=True))
te = time.time()
print("Deflated CG (vague attempt at recycling) time taken: ", (te-ts)*1000)

recycled_sol = []
#vector_factory = RitzFactory(subset_evaluator=RitzApproxKrylov())
#vector_factory = RitzFactory(subset_evaluator=RitzApriori(Bound=BoundCG))
vector_factory = RitzFactorySimple(n_vectors=k, which='sm')
ts = time.time()
#U=find_deflation_subspace(Asys[0],b,10)
#recycled_sol.append(DeflatedCg(systems[0],maxiter=1000))
#recycler = RecyclingCg(recycled_sol[0],vector_factory=vector_factory)
recycler = RecyclingCg(vector_factory=vector_factory)
for i in range(0,len(Asys)):
    recycled_sol.append(recycler.solve(systems[i],maxiter=1000))
te = time.time()
print("Recycled CG time taken: ", (te-ts)*1000)

# plot residuals
from matplotlib import pyplot
pyplot.figure(figsize=(6, 4), dpi=100)
pyplot.xlabel('Iteration $i$')
pyplot.ylabel(r'Relative residual norm $\frac{\|r_i\|}{\|b\|}$')
pyplot.semilogy(cg_sol[numSystems-1].resnorms,label='cg')
pyplot.semilogy(deflated_sol[numSystems-1].resnorms,label='deflated')
#pyplot.semilogy(defsingle[numSystems-1].resnorms,label='defsingle')
pyplot.semilogy(recycled_sol[numSystems-1].resnorms,label='recycled')
pyplot.legend()
pyplot.show()
