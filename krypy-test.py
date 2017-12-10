# -*- coding: utf-8 -*-
import time
import numpy
from krypy.linsys import LinearSystem, Gmres, Cg
from krypy.deflation import DeflatedCg, DeflatedGmres, Ritz
from krypy.utils import Arnoldi, ritz
from scipy import random, linalg

def find_deflation_subspace(A,b,k,ortho='dmgs',ritz_type='ritz'):
    Ar = Arnoldi(A,b,ortho=ortho)
    for i in range(1,k+1):
        Ar.advance()
    [V,H] = Ar.get()
    [theta,U,resnorm,Z] = ritz(H,V,type=ritz_type)
    return Z

matrixSize = 100
R = random.rand(matrixSize,matrixSize)
A = numpy.dot(R,R.transpose())
b=numpy.ones((matrixSize, 1))
linear_system = LinearSystem(A=A,b=b,self_adjoint=True,positive_definite=True)

ts = time.time()
cg_sol = Cg(linear_system,maxiter=1000)
te = time.time()
print("CG time taken: ", (te-ts)*1000)

ts = time.time()
deflated_sol = DeflatedCg(linear_system,U=find_deflation_subspace(A,b,10),maxiter=1000)
te = time.time()
print("Deflated CG time taken: ", (te-ts)*1000)

# plot residuals
from matplotlib import pyplot
pyplot.figure(figsize=(6, 4), dpi=100)
pyplot.xlabel('Iteration $i$')
pyplot.ylabel(r'Relative residual norm $\frac{\|r_i\|}{\|b\|}$')
pyplot.semilogy(cg_sol.resnorms,label='cg')
pyplot.semilogy(deflated_sol.resnorms,label='deflated')
pyplot.legend()
pyplot.show()
