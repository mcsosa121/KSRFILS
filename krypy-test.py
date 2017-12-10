# -*- coding: utf-8 -*-
import time
import numpy
from krypy.linsys import LinearSystem, Gmres, Cg
from krypy.deflation import DeflatedCg, DeflatedGmres, Ritz
from krypy.utils import Arnoldi, ritz

#A=numpy.diag([1e-3]+list(range(2, 101)))
from scipy import random, linalg
matrixSize = 100
R = random.rand(matrixSize,matrixSize)
A = numpy.dot(R,R.transpose())

b=numpy.ones((matrixSize, 1))
linear_system = LinearSystem(A=A,b=b,self_adjoint=True,positive_definite=True)

ts = time.time()
sol2 = Cg(linear_system,maxiter=1000)
te = time.time()
print((te-ts)*1000)
#sol3 = Gmres(linear_system)

# plot residuals
from matplotlib import pyplot

# use tex
#from matplotlib import rc
#rc('text', usetex=True)

# use beautiful style
#from mpltools import style
#style.use('ggplot')

pyplot.figure(figsize=(6, 4), dpi=100)
pyplot.xlabel('Iteration $i$')
pyplot.ylabel(r'Relative residual norm $\frac{\|r_i\|}{\|b\|}$')
#pyplot.semilogy(sol.resnorms)
pyplot.semilogy(sol2.resnorms,label='cg')
#pyplot.semilogy(sol3.resnorms)

#pyplot.savefig('example.png', bbox_inches='tight')


ts = time.time()
Ar = Arnoldi(A,b)
for i in range(1,11):
    Ar.advance()
[V,H] = Ar.get()
[theta,U,resnorm,Z] = ritz(H,V)
#sol = DeflatedGmres(linear_system,maxiter=1000,store_arnoldi=True,tol=1e-1)
#U = Ritz(sol, mode='harmonic')
sol4 = DeflatedCg(linear_system,U=Z,maxiter=1000)
#sol4 = DeflatedCg(linear_system,U=U.get_vectors(),maxiter=1000)
te = time.time()
print((te-ts)*1000)
pyplot.semilogy(sol4.resnorms,label='deflated')
pyplot.legend()
pyplot.show()
