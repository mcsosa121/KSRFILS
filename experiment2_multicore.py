import os
import time
import multiprocessing
from numpy import array, ones, dot
from scipy import linalg, io, rand, sparse
from krypy.linsys import LinearSystem, Cg
from krypy.deflation import DeflatedCg, DeflatedGmres, Ritz
from krypy.utils import Arnoldi, ritz, BoundCG, ConvergenceError, ArgumentError
from krypy.recycling import RecyclingCg
from krypy.recycling.factories import RitzFactory, RitzFactorySimple
from krypy.recycling.evaluators import RitzApriori, RitzApproxKrylov

import matplotlib.pyplot as plt
import pickle


def find_deflation_subspace(A, b, k, ortho='dmgs', ritz_type='ritz'):
    """
        Finds the deflation subspace for the given problem
    """
    Ar = Arnoldi(A, b, ortho=ortho)
    for i in range(1, k+1):
        Ar.advance()
    [V, H] = Ar.get()
    [theta, U, resnorm, Z] = ritz(H, V, type=ritz_type)
    return Z


def reuse_deflation_subspace(sol, ritz_type='ritz'):
    """ Returns a deflation subspace for later use """
    [theta, U, resnorm, Z] = ritz(sol.H, sol.V, type=ritz_type)
    return Z


def experiment(A, b=None, k=10, numSystems=5, rank=1, maxiter=1000):
    """ run experiment with A,b,k, numsystems, rank up to maxiterations """
    #form systems
    matrixSize = A.shape[0]
    if b == None:
        b = ones((matrixSize, 1))

    Asys = [A]
    for i in range(1, numSystems):
        u = rand(matrixSize, rank)
        Asys.append(Asys[i-1] + dot(u, u.T))

    systems = []
    for i in range(0, len(Asys)):
        systems.append(LinearSystem(A=Asys[i], b=b, self_adjoint=True, positive_definite=True))

    cg_sol = []
    ts = time.time()
    for i in range(0, len(Asys)):
        cg_sol.append(Cg(systems[i], maxiter=maxiter))
    te = time.time()
    cg_time = (te-ts)*1000

    deflated_sol = []
    ts = time.time()
    for i in range(0, len(Asys)):
        U = find_deflation_subspace(Asys[i], b, k)
        deflated_sol.append(DeflatedCg(systems[i], U=U, maxiter=maxiter))
    te = time.time()
    deflated_time = (te-ts)*1000

    recycled_sol = []
    vector_factory = RitzFactorySimple(n_vectors=k, which='sm')
    ts = time.time()
    recycler = RecyclingCg(vector_factory=vector_factory)
    for i in range(0, len(Asys)):
        recycled_sol.append(recycler.solve(systems[i], maxiter=maxiter))
    te = time.time()
    recycled_time = (te-ts)*1000

    return [cg_sol, deflated_sol, recycled_sol, cg_time, deflated_time, recycled_time]

def run_deflation(A, iterations, numSystems, filename, badq):
    """ Runs deflation solver on system Ax=b """
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

if __name__ == '__main__':
    run_tests('./up_iter/',15000,60)
