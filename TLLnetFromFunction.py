from TLLnetIO import TLLnet
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
import time

shared = None

class TLLnetFromFunction(TLLnet):

    def __init__(self, fn: Callable[[np.ndarray],np.ndarray], eta=0.01, polytope=None, tol=1e-9, rTol=1e-9, NUM_CPUS=4, NUM_GPUS=1, MULTI_CHUNK_SIZE=100, BUFFER_DEPTH=2):
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

        polytope = [polytope[0].copy(), polytope[1].copy().reshape(polytope[1].shape[0],-1)]

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
        self.numConstraints = self.H.shape[0]

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
        self.constraintBuffer = [ mp.RawArray('i', (self.H.shape[0]+1) * (self.MULTI_CHUNK_SIZE+1)) for ii in range(self.NUM_CPUS * self.BUFFER_DEPTH) ]
        self.bufferFreeQueue = mp.Queue()
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
            'constraintBuffer': self.constraintBuffer, \
            'n': self.n, \
            'numConstraints': self.numConstraints, \
            'MULTI_CHUNK_SIZE': self.MULTI_CHUNK_SIZE, \
            'sliceDim': self.sliceDim, \
            'bbox': self.bbox \
        }
        p = mp.Pool(self.NUM_CPUS,initializer=initPoolContext,initargs=(poolGlobals,))

        self.gpuWorkQueue = mp.Queue()
        self.gpuWorkers = [ \
                    mp.Process( \
                        target=fnsFromSlice, \
                        args=( \
                              ii, \
                              self.constraintBuffer, \
                              self.HShared, \
                              self.sliceDim, \
                              self.lb, \
                              self.spill, \
                              self.eta, \
                              self.dim0cnt, \
                              self.numConstraints, \
                              self.n, \
                              self.MULTI_CHUNK_SIZE, \
                              self.gpuWorkQueue, \
                              self.bufferFreeQueue \
                        ) \
                    ) for ii in range(self.NUM_GPUS) ]
        for pr in self.gpuWorkers:
            pr.start()


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
                        min(self.MULTI_CHUNK_SIZE , self.dim0cnt - chunkIdx), \
                        self.eta, \
                        self.bufferDoneQueue \
                    ) \
                )
            #r.get()
        while self.numGPUScheduled < self.dim0cnt:
            #print(self.sequentialChunks)
            self.scheduleSequentialChunks()
        p.close()
        p.join()

        # Wait for all of the GPU workers to finish
        while not self.gpuWorkQueue.empty():
            time.sleep(0.1)
        # Now shutdown all of the gpu workers
        for ii in range(self.NUM_GPUS):
            self.gpuWorkQueue.put([])
        for ii in range(self.NUM_GPUS):
            self.gpuWorkers[ii].join()
        print(np.frombuffer(self.constraintBuffer[0],dtype=np.int32))

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
                # Schedule this sequential chunk on a GPU:
                self.gpuWorkQueue.put(self.sequentialChunks[runIdx])
                # Now release the buffers
                #for res in self.sequentialChunks[runIdx]: self.bufferFreeQueue.put(res[0])

def initPoolContext(poolGlobals):
    global shared
    shared = poolGlobals
    lpObj = encapsulateLP.encapsulateLP()

def sliceBoundary(bufferIdx, chunkIdx, sliceLB, chunkLen, eta, myqueue):
    print(f'Enqueueing result on process {os.getpid()}')
    # Shortcut local variables for read-only shared globals:
    lp = encapsulateLP.encapsulateLP()
    N = shared['numConstraints']
    n = shared['n']
    sliceDim = shared['sliceDim']
    MULTI_CHUNK_SIZE = shared['MULTI_CHUNK_SIZE']
    bbox = shared['bbox']

    # This is the shared memory we will use to return the minimal set of constraints for the current slice:
    myBuffer = np.frombuffer(shared['constraintBuffer'][bufferIdx],dtype=np.int32).reshape((N+1, MULTI_CHUNK_SIZE+1))
    # Erase buffer with 0's
    np.copyto(myBuffer, np.zeros(myBuffer.shape,dtype=np.int32))

    # Create a local copy of the constraints from the shared memory
    H = np.frombuffer(shared['H'],dtype=np.float64).reshape((N, n+1)).copy()

    for ii in range(chunkLen+1):
        #print(f'[[PID {os.getpid()}]] {sliceLB + ii * eta}')
        if n > 1:
            if sliceLB + ii*eta > bbox[sliceDim,0] and sliceLB + ii*eta < bbox[sliceDim,1]:
                # Get the minimal Hrep of the n-1 dimensional slice
                temp = lpMinHRep( \
                                np.hstack([ \
                                                (H[:,0] + H[:,1+sliceDim]*(sliceLB + ii*eta)).reshape(N,-1), \
                                                (H[:, 1:(1+sliceDim)]).reshape(N,-1), \
                                                (H[:, (1+sliceDim+1):]).reshape(N,-1) \
                                            ]), \
                                None, \
                                range(H.shape[0]), \
                                lpObj=lp \
                            )
                myBuffer[:len(temp),ii] = np.array(temp,dtype=np.int32)
                myBuffer[-1,ii] = len(temp)
                print(f'[[PID {os.getpid()}]] slice coordinate = {sliceLB + ii * eta}; temp = {temp}')
        else:
            myBuffer[:N,ii] = np.arange(N,dtype=np.int32)
            myBuffer[-1,ii] = N
            print(f'[[PID {os.getpid()}]] slice coordinate = {sliceLB + ii * eta}; {myBuffer}')
    myqueue.put_nowait((bufferIdx, chunkIdx))
    return True

def fnsFromSlice(deviceId,constraintBuffer,HShared,sliceDim,lb,spill,eta,dim0cnt,N,n,MULTI_CHUNK_SIZE,inputQueue,constraintBufferQueue):
    buffers = [np.frombuffer(constraintBuffer[ii],dtype=np.int32).reshape((N+1, MULTI_CHUNK_SIZE+1)) for ii in range(len(constraintBuffer))]
    H = np.frombuffer(HShared,dtype=np.float64).reshape((N,n+1)).copy()
    results = []

    while True:
        # Should be a GPU_RATIO-length set of chunks
        workItem = inputQueue.get()
        if len(workItem) == 0:
            print(f'[[*GPU* PID {os.getpid()}]] EXITING!!')
            break
        workItem = sorted(workItem, key=lambda tup: tup[1])
        print(f'[[*GPU* PID {os.getpid()}]] Got gpu work item {workItem}')
        for it in workItem:
            print(f'[[*GPU* PID {os.getpid()}]] Buffer lookup for {it}: {buffers[it[0]]}')
            endSlice = min(workItem[-1][0], dim0cnt)
        sliceList = [ lb - spill + (it[1] + jj) * eta for it in workItem for jj in range(MULTI_CHUNK_SIZE) ]
        sliceList.append( lb - spill + (workItem[-1][1] + MULTI_CHUNK_SIZE ) * eta )
        sliceList = sliceList[:min(dim0cnt+1 - workItem[0][1],len(workItem)*MULTI_CHUNK_SIZE+1)]
        bdConstraints = np.zeros((buffers[workItem[0][0]].shape[0], len(workItem) * MULTI_CHUNK_SIZE + 1))
        for idx in range(len(workItem)):
            np.copyto(bdConstraints[:,1+(idx*MULTI_CHUNK_SIZE):1+((idx+1)*MULTI_CHUNK_SIZE)], buffers[workItem[idx][0]][:,1:])
        bdConstraints[:,0] = buffers[workItem[0][0]][:,0]
        print(f'[[*GPU* PID {os.getpid()}]] Slice list: {sliceList}')
        print(f'[[*GPU* PID {os.getpid()}]] Assembled bd constraints: {bdConstraints}')

        # Once work is done, release the associated constraint buffers
        for res in workItem:
            constraintBufferQueue.put(res[0])
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
