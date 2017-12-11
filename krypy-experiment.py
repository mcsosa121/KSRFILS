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

# # Time stuff
# class TimeoutException(Exception):
#     pass

# def timeout_handler(signum, frame):
#     raise TimeoutException

# signal.signal(signal.SIGALRM, timeout_handler)

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

'''
matrixSize = 100
numSystems = 30
R = random.rand(matrixSize,matrixSize)
A = numpy.dot(R,R.transpose())
[cg_sol,deflated_sol,recycled_sol,cg_time,deflated_time,recycled_time] = experiment(A,numSystems=numSystems)
print('CG time:',cg_time,'Deflated CG time:',deflated_time,'Recycled CG time:',recycled_time)
# plot residuals
from matplotlib import pyplot
pyplot.figure(figsize=(6, 4), dpi=100)
pyplot.xlabel('Iteration $i$')
pyplot.ylabel(r'Relative residual norm $\frac{\|r_i\|}{\|b\|}$')
pyplot.semilogy(cg_sol[numSystems-1].resnorms,label='cg')
pyplot.semilogy(deflated_sol[numSystems-1].resnorms,label='deflated')
pyplot.semilogy(recycled_sol[numSystems-1].resnorms,label='recycled')
pyplot.legend()
pyplot.show()
'''

def run_deflation(A,iterations,numSystems,filename,badq):
    try:
        [cg_sol,deflated_sol,recycled_sol,cg_time,deflated_time,recycled_time] = experiment(A,numSystems=numSystems,k=3,maxiter=iterations)
        itemlist = [filename,cg_sol[numSystems-1].resnorms,deflated_sol[numSystems-1].resnorms,recycled_sol[numSystems-1].resnorms,cg_time,deflated_time,recycled_time]
        with open('./results/res_'+filename[:-4]+'.txt', 'wb') as fp:
            pickle.dump(itemlist, fp)
    except (ConvergenceError, ArgumentError):
        badq.put(filename)


# TODO figure out why its failing whenever something passes
def run_tests(dir_name, iterations, timeout):
    directory = os.fsencode(dir_name)
    sizes = []
    cg_t = []
    deflated_t = []
    recycled_t = []
    numSystems=5

    dirl = os.listdir(directory)
    dirl.sort()

    badfiles = []
    timeoutfiles = []

    for file in dirl:
        filename = os.fsdecode(file)
        print(filename)
        Mat = io.loadmat(dir_name+filename)
        A = None
        for i in range(0,len(Mat['Problem'][0][0])):
            if type(Mat['Problem'][0][0][i]) == sparse.csc.csc_matrix:
                A = Mat['Problem'][0][0][i]
                break

        if A==None:
            print('Could not find matrix of',filename)
            continue
        else:
            # Starting the process
            badfq = multiprocessing.Queue()
            proc = multiprocessing.Process(
                target=run_deflation,
                name="run_deflation",
                args=(A,iterations,numSystems,filename,badfq)
                )
            proc.start()

            proc.join(timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                timeoutfiles.append(filename)
            else:
                badfiles.append(badfq.get())
                badfq.close()   

    print("Timed Out")
    print(timeoutfiles)
    print("Bad Files")
    print(badfiles)

'''
sizes = numpy.array(sizes)
cg_t = numpy.array(cg_t)
deflated_t = numpy.array(deflated_t)
recycled_t = numpy.array(recycled_t)
inds = sizes.argsort()
plt.plot(sizes[inds],cg_t[inds],label='cg')
plt.plot(sizes[inds],deflated_t[inds],label='deflated')
plt.plot(sizes[inds],recycled_t[inds],label='recycled')
plt.ylabel('time')
plt.xlabel('size of matrix')
plt.legend()
plt.show()
'''

if __name__ == '__main__':
    run_tests('./up_iter/',15000,60)
