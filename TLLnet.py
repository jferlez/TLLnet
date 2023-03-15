
import numpy as np

import scipy.special
import scipy

import math
import re
from functools import reduce
from copy import copy, deepcopy
import re
import pickle

npDataType = np.float64

numbaAvailable = False
try:
    # Numba imports
    import numba as nb
    from numba import njit
    from numba.core import types
    from numba.typed import Dict
    from numba.np.unsafe.ndarray import to_fixed_tuple
    from numba.pycc import CC
    selectorSetType = nb.typeof( \
                            nb.typed.List([ \
                                nb.typed.List([np.array([1,2], dtype=np.int64) for k in range(2)]) \
                                for _ in range(2) \
                            ]) \
                        )
    numbaAvailable = True
except ImportError:
    print('Warning (TLLnet): Numba is unavailable. \'evalAt\' method will be unaccelerated.')
    numbaAvailable = False

class TLLnet:

    props = ['n','N','M','m','localLinearFns','selectorSets','TLLFormatVersion']
    constructorArgs = {'input_dim':'n', 'output_dim':'m', 'linear_fns':'N', 'uo_regions':'M'}

    def __init__(self, input_dim=1, output_dim=1, linear_fns=1, uo_regions=None ):

        self.dtype = npDataType

        assert type(input_dim) == int and input_dim >= 1 , 'input_dim must be an integer >=1.'
        assert type(output_dim) == int and output_dim >= 1 , 'output_dim must be an integer >=1.'
        assert type(linear_fns) == int and linear_fns >= 1 , 'linear_fns must be an integer >=1.'
        assert (uo_regions is None) or (type(uo_regions) == int and uo_regions >= 1) , 'uo_regions must be None or an integer >=1.'

        self.TLLFormatVersion = '0.1.0'
        self.n=input_dim
        self.m=output_dim
        self.N=linear_fns
        if uo_regions is None:
            self.M = int(np.sum([scipy.special.binom((self.N*self.N-self.N)/2,i) for i in range(0,self.n+1)]))
        else:
            self.M=uo_regions

        self.model = None

        self.localLinearFns = [[np.zeros((self.N,self.n),dtype=self.dtype),np.zeros((self.N,),dtype=self.dtype)] for k in range(self.m)]
        if numbaAvailable:
            self.localLinearFnsNumbaKern = np.vstack([self.localLinearFns[out][0] for out in range(self.m)])
            self.localLinearFnsNumbaBias = np.vstack([self.localLinearFns[out][1].reshape(-1,1) for out in range(self.m)]).reshape(-1,1)
            self.localLinearFns = [[ \
                        self.localLinearFnsNumbaKern[(out*self.N):((out+1)*self.N),:], \
                        self.localLinearFnsNumbaBias[(out*self.N):((out+1)*self.N),0]  \
                    ] for out in range(self.m)]
        self.selectorSets = [[frozenset([0])] for k in range(self.m)]
        if numbaAvailable:
            self.selectorSetsNumba = nb.typed.List([ \
                    nb.typed.List([ (out * self.N) + np.array(list(i), dtype=np.int64) for i in self.selectorSets[out] ])
                    for out in range(self.m)
                ])

    def getLocalLinearFns(self):
        return self.localLinearFns

    def setLocalLinearFns(self,localLinearFns):

        assert len(localLinearFns) == self.m, 'Local linear functions must be specified for each output!'
        for k in range(self.m):
            assert localLinearFns[k][0].shape == (self.N,self.n) and localLinearFns[k][1].shape == (self.N,), 'Incorrect shape of local linear functions for output ' + str(k) + '!'


        # self.localLinearFns = [[np.array(x[0],dtype=self.dtype), np.array(x[1],dtype=self.dtype)] for x in localLinearFns]
        for out in range(self.m):
            for i in range(2):
                np.copyto(self.localLinearFns[out][i], localLinearFns[out][i])
        if self.model is not None:
            for k in range(self.m):
                self.setKerasLocalLinFns(self.localLinearFns[k][0].T, self.localLinearFns[k][1], out=k)
        
        if numbaAvailable:
            self.localLinearFnsNumbaKern = np.vstack([self.localLinearFns[out][0] for out in range(self.m)])
            self.localLinearFnsNumbaBias = np.vstack([self.localLinearFns[out][1].reshape(-1,1) for out in range(self.m)]).reshape(-1,1)
            self.localLinearFns = [[ \
                        self.localLinearFnsNumbaKern[(out*self.N):((out+1)*self.N),:], \
                        self.localLinearFnsNumbaBias[(out*self.N):((out+1)*self.N),0]  \
                    ] for out in range(self.m)]

    def getSelectorSets(self):
        return self.selectorSets

    def setSelectorSets(self,selectorSets):

        assert len(selectorSets) == self.m, 'Selector sets must be specified for each output!'
        for k in range(self.m):
            assert len(selectorSets[k]) <= self.M, 'Too many selector sets specified for output ' + str(k) + '!'


        self.selectorSets = deepcopy(selectorSets)
        if self.model is not None:
            for k in range(self.m):
                sIdx = 0
                for j in range(self.M):
                    self.setKerasSelector(self.selectorMatKerasFromSet(self.selectorSets[k][sIdx]), j, out=k )
                    if sIdx < len(self.selectorSets[k])-1:
                        sIdx += 1
        if numbaAvailable:
            self.selectorSetsNumba = nb.typed.List([ \
                    nb.typed.List([ (out * self.N) + np.array(list(i), dtype=np.int64) for i in self.selectorSets[out] ])
                    for out in range(self.m)
                ])

    def pointEval(self,pt):
        localEval = [ lf[0] @ pt.reshape(self.n,1) + lf[1].reshape(self.N,1) for lf in self.localLinearFns ]
        for out in range(len(self.selectorSets)):
            localEval[out] = np.max(np.array([ np.min(localEval[out][tuple(sSet),]) for sSet in self.selectorSets[out] ],dtype=npDataType))
        return np.array(localEval,dtype=npDataType)

    def activeLinearFunction(self,pt):
        localEval = [ lf[0] @ pt.reshape(self.n,1) + lf[1].reshape(self.N,1) for lf in self.localLinearFns ]
        fnsOut = []
        for out in range(len(self.selectorSets)):
            temp = np.argmax(np.array([ np.min(localEval[out][tuple(sSet),]) for sSet in self.selectorSets[out] ],dtype=npDataType))
            sSet = tuple(self.selectorSets[out][temp])
            fnsOut.append(sSet[ \
                               np.argmin(localEval[out][sSet,]) \
                               ])
        return fnsOut
    
    def evalAt(self, inPts, validateInput=True):
        if validateInput:
            assert type(inPts) is np.ndarray and len(inPts.shape) <= 2 and len(inPts.shape) > 0, f'Input must be a Numpy array with no more than two dimensions'
            assert inPts.shape[0] == self.n, f'Input dimension 0 must match number of TLL inputs n={self.n}'
            if len(inPts.shape) == 1:
                pts = inPts.reshape(-1,1)
            else:
                pts = inPts
        else:
            pts = inPts

        if numbaAvailable:
            return getActiveAtHelperNumba(self.n, self.N, self.M, self.m, self.localLinearFnsNumbaKern, self.localLinearFnsNumbaBias, self.selectorSetsNumba, pts)
        else:
            localEval = [ lf[0] @ pts.reshape(self.n, pts.shape[1]) + lf[1].reshape(self.N,1) for lf in self.localLinearFns ]
            fnsOut = np.zeros((self.m, pts.shape[1]), dtype=np.int64)
            minTerms = np.zeros((self.m, pts.shape[1]), dtype=np.int64)
            retEval = np.zeros((self.m, pts.shape[1]), dtype=np.float64)
            for out in range(len(self.selectorSets)):
                minTerms[out, :] = np.argmax(np.array([ np.min(localEval[out][tuple(sSet),],axis=0) for sSet in self.selectorSets[out] ],dtype=npDataType), axis=0)
                sSets = [ tuple(self.selectorSets[out][k]) for k in minTerms[out] ]
                fnsOut[out, :] = np.array( \
                        [ sSets[idx][ np.argmin(localEval[out][sSets[idx],idx], axis=0) ] for idx in range(len(sSets)) ], \
                        dtype=np.int64 \
                    )
                retEval[out, :] = np.array( \
                                [localEval[out][fnsOut[out,i], i] for i in range(fnsOut.shape[1])], dtype=np.float64 \
                            )
            return retEval, fnsOut, minTerms
    
    def selectorMatKerasFromSet(self,actSet):
        if len(actSet) == 0:
            raise ValueError('Please specify a non-empty set!')
        e = np.eye(self.N,dtype=self.dtype)
        ret = np.zeros((self.N,self.N),dtype=self.dtype)
        insertCounter = 0
        for i in actSet:
            ret[:,insertCounter] = e[:,i]
            insertCounter += 1
        for i in range(insertCounter,len(e)):
            ret[:,i] = ret[:,insertCounter-1]

        return ret

    def generateRandomCPWA(self,scale=1.0, ext=np.array([-10,10]), iterations=100000):
        if len(ext.shape)==1:
            ext = np.vstack([ext for i in range(self.n)])

        self.localLinearFns = []
        self.selectorSets = []

        for out in range(self.m):
            kern = np.random.normal(loc=0, scale=scale/10, size=(self.n, self.N)).astype(self.dtype)
            bias = np.random.normal(loc=0, scale=scale, size=(self.N,)).astype(self.dtype)

            idxs = np.array([ i for i in range(self.N) ], dtype=self.dtype)

            selMats = [[] for i in range(self.M)]
            selSets = [set([]) for i in range(self.M)]
            selSets[0] = intToSet( myRandSet(self.N) )
            selMats[0] = self.selectorMatKerasFromSet(selSets[0])

            matCounter = 1
            itCnt = iterations
            while matCounter < self.M and itCnt > 0:
                candidateSet = intToSet( myRandSet(self.N) )
                valid=True
                for k in range(matCounter):
                    if candidateSet.issubset(selSets[k]) or selSets[k].issubset(candidateSet) or selSets[k]==candidateSet:
                        valid=False
                        break
                if valid:
                    selSets[matCounter] = candidateSet
                    selMats[matCounter] = self.selectorMatKerasFromSet(candidateSet)
                    matCounter += 1
                itCnt -= 1

            intersections = [[] for k in range(matCounter)]
            for k in range(matCounter):
                intersections[k], resid, rank, singVals = np.linalg.lstsq( \
                        (kern @ selMats[k])[0:len(selSets[k]),:].T, \
                        (-(bias.reshape((1,len(bias))) @ selMats[k])[0:len(selSets[k])]).flatten() \
                    )
                intersections[k] = np.pad(intersections[k],(0,self.n),'maximum')[0:self.n]
            intersections = np.array(intersections)
            for k in range(matCounter,self.M):
                selMats[k] = selMats[matCounter-1]
            # kern = np.diag(0.5*(ext[out][1]-ext[out][0])/np.max(np.abs(intersections),axis=0)) @ kern
            kern = scale * (np.max(np.abs(intersections))) * kern
            self.localLinearFns.append([kern.T,bias])
            self.selectorSets.append(selSets[0:matCounter])
            if self.model is not None:
                self.setKerasLocalLinFns(kern,bias,out=out)
                for k in range(self.M):
                    self.setKerasSelector(selMats[k],k,out=out)
            # print(intersections)



    def toPythonIntSelectorSets(self):
        for k in range(self.m):
            for j in range(len(self.selectorSets[k])):
                oldSelectorSet = self.selectorSets[k][j]
                self.selectorSets[k][j] = set(list(map(lambda x: int(x),list(oldSelectorSet))))
                assert self.selectorSets[k][j] == oldSelectorSet

    def save(self, fname=None):
        saveDict = {}
        for p in self.props:
            saveDict[p] = getattr(self,p)
        if fname is not None:
            with open(fname,'wb') as fp:
                pickle.dump(saveDict,fp)
        return saveDict

    @classmethod
    def fromTLLFormatDict(cls, tllfile, validateFile=True):
        if type(tllfile) != dict:
            with open(tllfile, 'rb') as fp:
                tllDict = pickle.load(fp)
        else:
            tllDict = tllfile
            tllfile = 'Input dictionary'
        props = copy(cls.props)
        if not all([p in tllDict for p in props]):
            raise(TypeError(f'{tllfile} does not contain a valid TLL Format. One or more properties are missing.'))
        dtype = npDataType
        props = ['n','N','M','m']
        if validateFile:
            for p in props:
                if type(tllDict[p]) != int or tllDict[p] < 0:
                    raise(TypeError(f'{tllfile} does not contain a valid TLL format. {p} should be an integer > 0.'))
            for p in ['localLinearFns', 'selectorSets']:
                if len(tllDict[p]) != tllDict['m']:
                    raise(TypeError(f'{tllfile} does not contain a valid TLL format. {p} should be a list of length {tllDict["m"]}'))

            dtype = None
            shp = {0:(tllDict['N'], tllDict['n']), 1:(tllDict['N'],) }
            for j in range(tllDict['m']):
                if type(tllDict['localLinearFns'][j]) != list or len(tllDict['localLinearFns'][j]) != 2:
                    raise(TypeError(f'{tllfile} does not contain a valid TLL format. Element {j} of \'localLinearFns\' should be a list of length 2.'))
                for k in [0,1]:
                    if type(tllDict['localLinearFns'][j][k]) != np.ndarray or tllDict['localLinearFns'][j][k].shape != shp[k]:
                        raise(TypeError(f'{tllfile} does not contain a valid TLL format. \'localLinearFns\' property should be a NumPy of shape {shp[k]}'))
                    if j == 0 and k == 0:
                        dtype = tllDict['localLinearFns'][j][k].dtype
                    if tllDict['localLinearFns'][j][k].dtype != dtype:
                        raise(TypeError(f'{tllfile} does not contain a valid TLL format. \'localLinearFns\' NumPy arrays should not be of different data types.'))
                if type(tllDict['selectorSets'][j]) != list or len(tllDict['selectorSets'][j]) == 0:
                    raise(TypeError(f'{tllfile} does not contain a valid TLL format. Element {j} of property \'selectorSets\' should be a list of length at least 1.'))
                for k in range(len(tllDict['selectorSets'][j])):
                    if type(tllDict['selectorSets'][j][k]) != set \
                                or any([(type(el) != int and type(el) != np.int64 and type(el) != np.uint64) for el in tllDict['selectorSets'][j][k]]) \
                                or min(tllDict['selectorSets'][j][k]) < 0 \
                                or max(tllDict['selectorSets'][j][k]) >= tllDict['N']:
                            raise(TypeError(f'{tllfile} does not contain a valid TLL format. Selector set {k} for output {j} should be a set of integers between 0 and {tllDict["N"]-1}'))
        tllDict['dtype'] = dtype
        return tllDict


    @classmethod
    def fromTLLFormat(cls, tllfile, validateFile=True):
        tllDict = cls.fromTLLFormatDict(tllfile, validateFile=validateFile)
        tll = cls(**{ky:tllDict[val] for ky, val in cls.constructorArgs.items() if val in tllDict})
        tll.dtype = tllDict['dtype']
        tll.setLocalLinearFns(tllDict['localLinearFns'])
        tll.setSelectorSets(tllDict['selectorSets'])

        return tll


def selectorMatrixToSet(sMat):
    return set(list(map(lambda x: int(x), np.nonzero(sMat)[0].tolist())))

def myRandSet(N):
    if N <= 63:
        intOut = int(np.random.randint(low = 1, high=((2**N)-1)))
    else:
        intOut = 0
        Ntemp = N
        while Ntemp > 0:
            if Ntemp > 63:
                size = 63
            else:
                size = Ntemp
            intOut = intOut << size
            intOut += int(np.random.randint(low = 1 if size > 1 else 0, high=((2**size)-1)))
            Ntemp -= size
    return intOut

def intToSet(input_int):
    output_list = []
    intCopy = input_int
    index = 0
    while intCopy > 0:
        if intCopy & 1 > 0:
            output_list.append(index)
        index += 1
        intCopy = intCopy >> 1
    return frozenset(output_list)

if numbaAvailable:
    @njit( \
        nb.types.Tuple((nb.types.float64[:,:], nb.types.int64[:,:], nb.types.int64[:,:]))( \
            nb.types.int64, \
            nb.types.int64, \
            nb.types.int64, \
            nb.types.int64, \
            nb.types.float64[:,::1], \
            nb.types.float64[:,:], \
            selectorSetType, \
            nb.types.float64[:,::1] \
        ), \
        cache=True \
    )
    def getActiveAtHelperNumba(n, N, M, m, localLinearFnsKern, localLinearFnsBias, selectorSets, pts):
        fnsOut = np.zeros((m, pts.shape[1]), dtype=np.int64)
        minTerms = np.zeros((m, pts.shape[1]), dtype=np.int64)
        retEval = np.full((m, pts.shape[1]), -np.inf, dtype=np.float64)
        nPts = pts.shape[1]

        # minTermVal = np.full((m,nPts),np.inf, dtype=np.float64)
        tempMinVal = np.full((nPts,),np.inf, dtype=np.float64)
        # maxTermVal = np.full((m,nPts),-np.inf, dtype=np.float64)
        # minIdx = np.zeros((m,nPts),dtype=np.int64)
        tempMinIdx = np.zeros((nPts,),dtype=np.int64)
        # maxIdx = np.zeros((m,nPts),dtype=np.int64)

        localEval = localLinearFnsKern @ pts + localLinearFnsBias


        for out in range(m):
            for sIdx in range(len(selectorSets[out])):
                # Compute one min term:
                tempMinVal[:] = np.full((nPts,),np.inf,dtype=np.float64)
                tempMinIdx[:] = np.zeros((nPts,),dtype=np.int64)
                for idx in selectorSets[out][sIdx]:
                    for pIdx in range(nPts):
                        if localEval[idx,pIdx] < tempMinVal[pIdx]:
                            tempMinVal[pIdx] = localEval[idx,pIdx]
                            tempMinIdx[pIdx] = idx % N
                # Update max as needed
                for pIdx in range(nPts):
                    if tempMinVal[pIdx] > retEval[out,pIdx]:
                        retEval[out,pIdx] = tempMinVal[pIdx]
                        fnsOut[out,pIdx] = tempMinIdx[pIdx]
                        minTerms[out,pIdx] = sIdx

        return retEval, fnsOut, minTerms


# if __name__ == '__main__':

#     # mxnet = MinMaxBankByN(numGroups=2,groupSize=3,outputDim=2)

#     # x = TLLnet(input_dim=1, output_dim=2, linear_fns=10)

#     # t2 = TLLnet(input_dim=1, output_dim=1, linear_fns=5, uo_regions=25)

#     tFlat = TLLnet(input_dim=3,output_dim=2,linear_fns=5,uo_regions=29,incBias=True,flat=True)

#     print('done')
