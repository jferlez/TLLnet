from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Layer
from keras import regularizers
from keras import initializers
from numpy.lib.function_base import select
import tensorflow as tf
from keras import backend as K
import keras
import numpy as np
import scipy.special
import scipy
import re
from functools import reduce

npDataType = np.float64
tfDataType = tf.float64
K.set_floatx('float64')


class TLLnet:

    def __init__(self, input_dim=1, output_dim=1, linear_fns=1, uo_regions=None, incBias=False, flat=False):
        self.n=input_dim
        self.m=output_dim
        self.N=linear_fns
        self.flat = flat
        self.incBias = incBias
        if uo_regions == None:
            self.M = int(np.sum([scipy.special.binom((self.N*self.N-self.N)/2,i) for i in range(0,self.n+1)]))
        else:
            self.M=uo_regions

        inlayer = Input(shape=(self.n,))

        linearLayer = Dense(self.N * self.m)
        selectorLayer = Dense(self.N*self.M*self.m,use_bias=incBias)

        x = selectorLayer(linearLayer(inlayer))

        minMaxStartLayer = 3
        if not flat:
            x = tf.keras.layers.Reshape( \
                    (self.m, self.M, self.N) if self.m > 1 else (self.M, self.N) \
                )( \
                    selectorLayer(linearLayer(inlayer)) \
                )
            minMaxStartLayer = 4
        
        lays = []

        reduc = self.N
        while reduc > 1:
            mm = MinMaxBankByN(self.M, reduc, self.m, maxQ=False, incBias=incBias,flat=flat)
            lays.append(mm)
            reduc = int(mm[1]/(self.m*self.M)) if flat else mm[1]
            x = mm[0][1](mm[0][0](x))

        if not flat:
            x = tf.keras.layers.Reshape( \
                    (self.m, self.M) if self.m > 1 else (self.M,) \
                )(x)
            lays.append(())

        reduc = self.M
        while reduc > 1:
            mm = MinMaxBankByN(groupSize=reduc,outputDim=self.m, incBias=incBias, flat=flat)
            lays.append(mm)
            reduc = int(mm[1]/self.m) if flat else mm[1]
            x = mm[0][1](mm[0][0](x))
        

        if not flat:
            x = tf.keras.layers.Reshape((self.m,))(x)
            

        self.model = Model(inputs=inlayer, outputs=x)

        for i in range(len(lays)):
            if len(lays[i]) == 0:
                # When we need to skip a Reshape layer, we need to add 1 to the 'effective' layer index.
                # Since we're going two *actual* layers at a time for each element in lays, the offset is -2+1 = -1.
                # (This is because we have the 2*i in the 'effective' layer index calculation.)
                minMaxStartLayer = minMaxStartLayer-1
                continue
            if incBias:
                self.model.layers[minMaxStartLayer+2*i].set_weights([lays[i][2],0.*self.model.layers[minMaxStartLayer+2*i].get_weights()[1]])
                self.model.layers[minMaxStartLayer+2*i+1].set_weights([lays[i][3],0.*self.model.layers[minMaxStartLayer+2*i+1].get_weights()[1]])
            else:
                self.model.layers[minMaxStartLayer+2*i].set_weights([lays[i][2]])
                self.model.layers[minMaxStartLayer+2*i+1].set_weights([lays[i][3]])
        
        if incBias:
            self.model.layers[2].set_weights( [self.model.layers[2].get_weights()[0], 0.*self.model.layers[2].get_weights()[1] ] )

        self.linearLayer = self.model.layers[1]
        self.selectorLayer = self.model.layers[2]
    
    def setLocalLinFns(self, kern, bias, out=0):
        currWeights = self.linearLayer.get_weights()

        currWeights[0][:, (out*self.N):((out+1)*self.N) ] = kern
        currWeights[1][ (out*self.N):((out+1)*self.N) ] = bias

        self.linearLayer.set_weights(currWeights)
    
    def getLocalLinFns(self, out=0):
        currWeights = self.linearLayer.get_weights()

        return [ \
                currWeights[0][:, (out*self.N):((out+1)*self.N) ], \
                currWeights[1][ (out*self.N):((out+1)*self.N) ] \
            ]
    
    def getAllLocalLinFns(self):

        return [ \
                self.getLocalLinFns(out=k) for k in range(self.m) \
            ]
    
    def setSelector(self, arr, idx, out=0):
        if idx >= self.M:
            raise ValueError('Specified index must be less than the number of UO Regions!')

        currWeights = self.selectorLayer.get_weights()

        currWeights[0][:, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ] = np.zeros((self.m*self.N,self.N))
        currWeights[0][out*self.N:(out+1)*self.N, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ] = arr

        self.selectorLayer.set_weights(currWeights)
    
    def setSelectorBroken(self, arr, idx, out=0):
        if idx >= self.M:
            raise ValueError('Specified index must be less than the number of UO Regions!')

        currWeights = self.selectorLayer.get_weights()

        currWeights[0][:, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ] = np.zeros((self.m*self.N,self.N))
        currWeights[0][:, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ] = arr

        self.selectorLayer.set_weights(currWeights)
    
    def getSelector(self, idx, out=0):
        if idx >= self.M:
            raise ValueError('Specified index must be less than the number of UO Regions!')

        currWeights = self.selectorLayer.get_weights()

        return currWeights[0][out*self.N:(out+1)*self.N, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ]
    
    def getAllSelectors(self):

        return [ \
                [self.getSelector(j, out=k) for j in range(self.M)] for k in range(self.m) \
            ]

    def selectorMatFromSet(self,actSet):
        if len(actSet) == 0:
            raise ValueError('Please specify a non-empty set!')
        e = np.eye(self.N)
        ret = np.zeros((self.N,self.N))
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
        
        for out in range(self.m):
            kern = np.random.normal(loc=0, scale=scale/10, size=(self.n, self.N))
            bias = np.random.normal(loc=0, scale=scale, size=(self.N,))
            
            idxs = np.array([ i for i in range(self.N) ])
            
            selMats = [[] for i in range(self.M)]
            selSets = [set([]) for i in range(self.M)]
            selSets[0] = intToSet( myRandSet(self.N) )
            selMats[0] = self.selectorMatFromSet(selSets[0])
            
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
                    selMats[matCounter] = self.selectorMatFromSet(candidateSet)
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
            self.setLocalLinFns(kern,bias,out=out)
            for k in range(self.M):
                self.setSelector(selMats[k],k,out=out)
            # print(intersections)
            

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

def MinMaxBankByN(numGroups=1,groupSize=2,outputDim=1,maxQ=True,incBias=False,flat=False):

    if groupSize==1:
        raise ValueError('groupSize argument must be >1 (no min/max needed otherwise!)')

    odd = np.mod(groupSize,2)==1

    if maxQ:
        finalWeights = np.array([0.5,-0.5,0.5,0.5],npDataType)
    else:
        finalWeights = np.array([0.5,-0.5,-0.5,-0.5],npDataType)
    

    if outputDim > 1:
        if numGroups > 1:
            shapeTuple = (outputDim,numGroups,groupSize)
        else:
            shapeTuple = (outputDim,groupSize)
    else:
        if numGroups > 1:
            shapeTuple = (numGroups,groupSize)
        else:
            shapeTuple = (groupSize,)

    if flat:
        shapeTuple = (reduce(lambda x, y: x*y, shapeTuple),)

    # inlayer = Input(shape=(outputDim*numGroups*groupSize,))

    numMins = int((groupSize-1)/2 if odd else groupSize/2)

    compLayerKernel = np.zeros((groupSize,4*(numMins + 1) if odd else 4*numMins),npDataType)

    for k in range(numMins):
        compLayerKernel[2*k:2*k+2, 4*k:4*(k+1)] = np.array([[1,1],[-1,-1],[-1,1],[1,-1]],npDataType).T
    # Compares the last two elements of the block, even though one of them alread
    # appears in another comparison
    if odd:
        compLayerKernel[groupSize-2:groupSize,compLayerKernel.shape[1]-4:compLayerKernel.shape[1]] = np.array([[1,1],[-1,-1],[-1,1],[1,-1]],npDataType).T
    
    outLayerKernel = np.zeros((compLayerKernel.shape[1], numMins + 1 if odd else numMins))

    for k in range(numMins+1 if odd else numMins):
        outLayerKernel[4*k:4*(k+1),k] = finalWeights

    # Make a block diagonal matrix; see https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
    if flat:
        compLayerKernel = np.kron(np.eye(numGroups*outputDim,dtype=int),compLayerKernel)
        outLayerKernel = np.kron(np.eye(numGroups*outputDim,dtype=int),outLayerKernel)

    compLayer = Dense(compLayerKernel.shape[1],use_bias=incBias,trainable=False,activation='relu',input_shape=shapeTuple)

    outLayer = Dense(outLayerKernel.shape[1],use_bias=incBias,trainable=False)

    return ((compLayer,outLayer),outLayerKernel.shape[1],compLayerKernel,outLayerKernel)






if __name__ == '__main__':
    
    # mxnet = MinMaxBankByN(numGroups=2,groupSize=3,outputDim=2)

    # x = TLLnet(input_dim=1, output_dim=2, linear_fns=10)

    # t2 = TLLnet(input_dim=1, output_dim=1, linear_fns=5, uo_regions=25)

    tFlat = TLLnet(input_dim=3,output_dim=2,linear_fns=5,uo_regions=29,incBias=True,flat=True)

    print('done')