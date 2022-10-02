from TLLnet import TLLnet
import encapsulateLP
from region_helpers import lpMinHRep, findInteriorPoint
import math
import numpy as np
import numba as nb
from typing import Callable
import jax
import jax.numpy as jnp
from jax import jit
import multiprocessing as mp
import queue
import os

shared = None

class TLLnetFromFunction(TLLnet):

    def __init__(self, fn: Callable[[np.ndarray],np.ndarray], eta=0.01, polytope=None, tol=1e-9, rTol=1e-9, NUM_CPUS=4, NUM_GPUS=0, MULTI_CHUNK_SIZE=100, BUFFER_DEPTH=2):
        self.eta = eta
        self.NUM_CPUS = NUM_CPUS
        self.NUM_GPUS = NUM_GPUS
        self.MULTI_CHUNK_SIZE = MULTI_CHUNK_SIZE
        self.BUFFER_DEPTH = BUFFER_DEPTH
        self.GPU_RATIO = max(self.NUM_CPUS//self.NUM_GPUS if self.NUM_GPUS > 0 else 1,1)
        self.tol = tol
        self.rTol = rTol
        # We will use this object for LP calls:
        self.lp = encapsulateLP.encapsulateLP()

        # Find an interior point of the specified polytope:
        self.pt = findInteriorPoint( np.hstack([-polytope[1],polytope[0]]), lpObj=self.lp )
        assert self.pt is not None, 'ERROR: specified polytope does not have an interior!'

        self.n = self.pt.flatten().shape[0]

        try:
            testOut = fn(self.pt)
        except:
            print(f'ERROR: Unable to compute function output for polytope point {self.pt}')
            raise ValueError

        self.m = testOut.flatten().shape[0]

        # Specify the polytope
        H = np.hstack([-polytope[1],polytope[0]])
        # Remove any redundant constraints:
        self.H = H[lpMinHRep(H,None,range(H.shape[0]),lpObj=self.lp),:]

    def generateLocalLinearFns(self):
        # Use a list, because we will add these in chunks of unknown size
        self.localLinearFns = []

        cpuBufferLoc = [ -1 for ii in range(self.NUM_CPUS) ]
        gpuBufferLoc = [ -1 for ii in range(self.NUM_CPUS) ]
        cpuFreeQueue = queue.Queue()
        gpuFreeQueue = queue.Queue()
        # define some shared memory
        # put H in some shared memory
        self.HShared = mp.RawArray('d', self.H.shape[0] * self.H.shape[1])
        self.HSharedNP = np.frombuffer( self.HShared, dtype=np.float64).reshape(self.H.shape)
        np.copyto(self.HSharedNP, self.H)

        # define shared memory for the return values from each pool worker
        self.constraintBuffer = [ mp.Array('i', self.H.shape[0] * self.MULTI_CHUNK_SIZE) for ii in range(self.NUM_CPUS * self.BUFFER_DEPTH) ]
        self.bufferFreeQueue = queue.Queue()
        for ii in range(len(self.constraintBuffer)):
            self.bufferFreeQueue.put(ii)
        self.sequentialChunks = {}

        self.bbox = constraintBoundingBox(self.H,lpObj=self.lp)
        self.sliceDim = 0
        self.lb = self.bbox[self.sliceDim,0]
        self.ub = self.bbox[self.sliceDim,1]
        self.dim0cnt = math.ceil( (self.ub - self.lb)/self.eta )
        self.spill = ( self.eta * self.dim0cnt - (self.ub - self.lb) )/2.0
        self.numGPUScheduled = 0

        mgr = mp.Manager()
        self.bufferDoneQueue = mgr.Queue()
        poolGlobals = { \
            'H': self.HShared, \
            'q': self.bufferDoneQueue, \
            'constraintBuffer': self.constraintBuffer \
        }
        p = mp.Pool(self.NUM_CPUS,initializer=initPoolContext,initargs=(poolGlobals,))

        for chunkIdx in range(0,self.dim0cnt,self.MULTI_CHUNK_SIZE):
            chunk = (self.lb - self.spill + chunkIdx * self.eta, self.lb - self.spill + min(chunkIdx + self.MULTI_CHUNK_SIZE , self.dim0cnt) * self.eta )
            print(chunk)
            while self.bufferFreeQueue.empty() or not self.bufferDoneQueue.empty():
                self.scheduleSequentialChunks()
            bufferIdx = self.bufferFreeQueue.get()
            r = p.apply_async( \
                    sliceBoundary, \
                    ( \
                        bufferIdx, \
                        chunkIdx, \
                        self.lb - self.spill + chunkIdx * self.eta, \
                        min(chunkIdx + self.MULTI_CHUNK_SIZE , self.dim0cnt), \
                        self.eta, \
                        self.bufferDoneQueue \
                    ) \
                )
        while self.numGPUScheduled < self.dim0cnt:
            #print(self.sequentialChunks)
            self.scheduleSequentialChunks()
        p.close()
        p.join()

    def scheduleSequentialChunks(self):
        runOnce = True
        while runOnce or not self.bufferDoneQueue.empty():
            print('Waiting on bufferDoneQueue')
            runOnce = False
            # Either we have something in bufferDoneQueue or else all the buffers are in use, so wait for one to be done
            result = self.bufferDoneQueue.get()
            print(f'Got result: {result}')
            runIdx = result[1]//(self.GPU_RATIO*self.MULTI_CHUNK_SIZE)
            if runIdx not in self.sequentialChunks:
                self.sequentialChunks[runIdx] = [result]
            else:
                self.sequentialChunks[runIdx].append(result)
            print(self.dim0cnt - runIdx*self.GPU_RATIO*self.MULTI_CHUNK_SIZE)
            endChunks = math.ceil((self.dim0cnt - runIdx*self.GPU_RATIO*self.MULTI_CHUNK_SIZE) / (1.0*self.MULTI_CHUNK_SIZE)) \
                        if (self.dim0cnt - runIdx*self.GPU_RATIO*self.MULTI_CHUNK_SIZE) < self.GPU_RATIO*self.MULTI_CHUNK_SIZE else self.GPU_RATIO
            endCnt = min(self.dim0cnt - runIdx*self.GPU_RATIO*self.MULTI_CHUNK_SIZE, self.GPU_RATIO*self.MULTI_CHUNK_SIZE)
            #print(f'Result {result} has endCnt = {endCnt} and endChunks = {endChunks};\nsequentialChunks = {self.sequentialChunks}')
            if len(self.sequentialChunks[runIdx]) == endChunks:
                # Schedule this sequential chunk on the first available GPU
                chunkBegin = min([x[1] for x in self.sequentialChunks[runIdx]])
                #print(f'Chunks {self.sequentialChunks[runIdx]} ready for GPU scheduling')
                print(f'Slice = ({self.lb - self.spill + chunkBegin *self.eta}, {self.lb - self.spill + min(self.dim0cnt, chunkBegin + self.MULTI_CHUNK_SIZE*self.GPU_RATIO)*self.eta})')
                self.numGPUScheduled += endCnt
                # Now release the buffers
                for res in self.sequentialChunks[runIdx]: self.bufferFreeQueue.put(res[0])

def initPoolContext(poolGlobals):
    global shared
    shared = poolGlobals

def sliceBoundary(bufferIdx, chunkIdx, sliceLB, chunkLen, eta , myqueue):
    print(f'Enqueueing result on process {os.getpid()}')
    myqueue.put_nowait((bufferIdx, chunkIdx))
    return True



def constraintBoundingBox(constraints,basis=None,lpObj=None):
    if lpObj is None:
        lpObj = encapsulateLP.encapsulateLP()
    solver = 'glpk'
    n = constraints.shape[1]-1
    if basis is None:
        bs = np.eye(n)
    else:
        bs = basis.copy()
    #if constraints[0].shape[1] != 2:
    #print(f'constraints = {constraints}')
    #print(type(constraints[0][0]))
    bboxIn = np.inf * np.ones((n,2),dtype=np.float64)
    bboxIn[:,0] = -bboxIn[:,0]
    ed = np.zeros((n,1))
    for ii in range(n):
        for direc in [1,-1]:
            status, x = lpObj.runLP( \
                direc * bs[ii,:], \
                -constraints[:,1:], constraints[:,0], \
                lpopts = {'solver':solver, 'fallback':'glpk'} if solver != 'glpk' else {'solver':'glpk'}, \
                msgID = 0 \
            )
            x = np.frombuffer(x)

            if status == 'optimal':
                bboxIn[ii,(0 if direc == 1 else 1)] = np.dot(x,bs[ii,:])
            elif status == 'dual infeasible':
                bboxIn[ii,(0 if direc == 1 else 1)] = -1*direc*np.inf
            else:
                print('********************  WARNING!!  ********************')
                print('Infeasible or numerical ill-conditioning detected while computing bounding box!')
                return bboxIn
    return bboxIn