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
from copy import deepcopy
import re
import pickle

npDataType = np.float64
tfDataType = tf.float64
# K.set_floatx('float64')

onnxAvailable = True
try:
    import tf2onnx
    import onnx
    import onnxruntime as rt
except ImportError:
    print('WARNING: tf2onnx or onnxruntime are unavailable. Exporting TLL to ONNX will be unavailable.')
    onnxAvailable = False

typeMismatchWarning = True

class TLLnet:

    def __init__(self, input_dim=1, output_dim=1, linear_fns=1, uo_regions=None, dtype=npDataType, dtypeKeras=tfDataType):
        global typeMismatchWarning
        self.dtype = dtype
        self.dtypeKeras = dtypeKeras

        if dtype != dtypeKeras and typeMismatchWarning:
            print('WARNING: TLL created with different internal datatype (' + str(dtype) + ') and Keras datatype (' + str(dtypeKeras) + '); this warning will not be repeated.')
            typeMismatchWarning = False

        if self.dtypeKeras == tf.float64:
            self.dtypeNpKeras = np.float64
        elif self.dtypeKeras == tf.float32:
            self.dtypeNpKeras = np.float32
        else:
            self.dtypeNpKeras = np.float64

        assert type(input_dim) == int and input_dim >= 1 , 'input_dim must be an integer >=1.'
        assert type(output_dim) == int and output_dim >= 1 , 'output_dim must be an integer >=1.'
        assert type(linear_fns) == int and linear_fns >= 1 , 'linear_fns must be an integer >=1.'
        assert (uo_regions is None) or (type(uo_regions) == int and uo_regions >= 1) , 'uo_regions must be None or an integer >=1.'

        self.n=input_dim
        self.m=output_dim
        self.N=linear_fns
        if uo_regions is None:
            self.M = int(np.sum([scipy.special.binom((self.N*self.N-self.N)/2,i) for i in range(0,self.n+1)]))
        else:
            self.M=uo_regions
        
        self.model = None

        self.localLinearFns = [[np.zeros((self.N,self.n)),np.zeros((self.N,))] for k in range(self.m)]
        self.selectorSets = [[frozenset([0])] for k in range(self.m)]

    def setLocalLinearFns(self,localLinearFns):

        assert len(localLinearFns) == self.m, 'Local linear functions must be specified for each output!'
        for k in range(self.m):
            assert localLinearFns[k][0].shape == (self.N,self.n) and localLinearFns[k][1].shape == (self.N,), 'Incorrect shape of local linear functions for output ' + str(k) + '!'
        

        self.localLinearFns = [[x[0].astype(dtype=self.dtype), x[1].astype(dtype=self.dtype)] for x in localLinearFns]
        if self.model is not None:
            for k in range(self.m):
                self.setKerasLocalLinFns(self.localLinearFns[k][0].T, self.localLinearFns[k][1], out=k)

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

    def createKeras(self, incBias=False, flat=False):
        self.flat = flat
        self.incBias = incBias
        self.inlayer = Input(shape=(self.n,), dtype=self.dtypeKeras)

        self.linearLayer = Dense(self.N * self.m, dtype=self.dtypeKeras, name='linearLayer')
        self.selectorLayer = Dense(self.N*self.M*self.m,use_bias=incBias, dtype=self.dtypeKeras, name='selectionLayer')

        x = self.selectorLayer(self.linearLayer(self.inlayer))

        self.minMaxStartLayer = 3
        if not flat:
            x = tf.keras.layers.Reshape( \
                    (self.m, self.M, self.N) if self.m > 1 else (self.M, self.N) \
                )( \
                    self.selectorLayer(self.linearLayer(self.inlayer)) \
                )
            self.minMaxStartLayer = 4
        
        self.lays = []

        bankIdx = 0
        reduc = self.N
        while reduc > 1:
            mm = MinMaxBankByN(self.M, reduc, self.m, maxQ=False, incBias=incBias,flat=flat, dtypeKeras=self.dtypeKeras, layerIdx=bankIdx)
            self.lays.append(mm)
            reduc = int(mm[1]/(self.m*self.M)) if flat else mm[1]
            x = mm[0][1](mm[0][0](x))
            bankIdx += 1

        if not flat:
            x = tf.keras.layers.Reshape( \
                    (self.m, self.M) if self.m > 1 else (self.M,), \
                    dtype=self.dtypeKeras \
                )(x)
            self.lays.append(())

        bankIdx = 0
        reduc = self.M
        while reduc > 1:
            mm = MinMaxBankByN(groupSize=reduc,outputDim=self.m, incBias=incBias, flat=flat, dtypeKeras=self.dtypeKeras, layerIdx=bankIdx)
            self.lays.append(mm)
            reduc = int(mm[1]/self.m) if flat else mm[1]
            x = mm[0][1](mm[0][0](x))
            bankIdx += 1
        

        if not flat:
            x = tf.keras.layers.Reshape((self.m,), dtype=self.dtypeKeras)(x)
        
        self.outputLayer = x

        self.model = Model(inputs=self.inlayer, outputs=self.outputLayer)

        for i in range(len(self.lays)):
            if len(self.lays[i]) == 0:
                # When we need to skip a Reshape layer, we need to add 1 to the 'effective' layer index.
                # Since we're going two *actual* layers at a time for each element in lays, the offset is -2+1 = -1.
                # (This is because we have the 2*i in the 'effective' layer index calculation.)
                minMaxStartLayer = minMaxStartLayer-1
                continue
            if incBias:
                self.model.layers[self.minMaxStartLayer+2*i].set_weights([self.lays[i][2],0.*self.model.layers[self.minMaxStartLayer+2*i].get_weights()[1]])
                self.model.layers[self.minMaxStartLayer+2*i+1].set_weights([self.lays[i][3],0.*self.model.layers[self.minMaxStartLayer+2*i+1].get_weights()[1]])
            else:
                self.model.layers[self.minMaxStartLayer+2*i].set_weights([self.lays[i][2]])
                self.model.layers[self.minMaxStartLayer+2*i+1].set_weights([self.lays[i][3]])
        
        if incBias:
            self.model.layers[2].set_weights( [self.model.layers[2].get_weights()[0], 0.*self.model.layers[2].get_weights()[1] ] )

        # self.linearLayer = self.model.layers[1]
        # self.selectorLayer = self.model.layers[2]

        for k in range(self.m):
            self.setKerasLocalLinFns(self.localLinearFns[k][0].T, self.localLinearFns[k][1],out=k)
            sIdx = 0
            for j in range(self.M):
                self.setKerasSelector(self.selectorMatKerasFromSet(self.selectorSets[k][sIdx]), j, out=k)
                if sIdx < len(self.selectorSets[k]) - 1:
                    sIdx += 1
    
    def setKerasLocalLinFns(self, kern, bias, out=0):
        currWeights = self.linearLayer.get_weights()

        currWeights[0][:, (out*self.N):((out+1)*self.N) ] = kern.astype(self.dtypeNpKeras)
        currWeights[1][ (out*self.N):((out+1)*self.N) ] = bias.astype(self.dtypeNpKeras)

        self.linearLayer.set_weights(currWeights)
    
    def getKerasLocalLinFns(self, out=0, transpose=False):
        currWeights = self.linearLayer.get_weights()

        retWeights = [ \
                currWeights[0][:, (out*self.N):((out+1)*self.N) ].astype(self.dtype), \
                currWeights[1][ (out*self.N):((out+1)*self.N) ].astype(self.dtype) \
            ]
        if transpose:
            retWeights[0] = retWeights[0].T
        return retWeights
    
    def getKerasAllLocalLinFns(self, transpose=False):

        return [ \
                self.getKerasLocalLinFns(out=k,transpose=transpose) for k in range(self.m) \
            ]
    
    def setKerasSelector(self, arr, idx, out=0):
        if idx >= self.M:
            raise ValueError('Specified index must be less than the number of UO Regions!')

        currWeights = self.selectorLayer.get_weights()

        currWeights[0][:, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ] = np.zeros((self.m*self.N,self.N),dtype=self.dtypeNpKeras)
        currWeights[0][out*self.N:(out+1)*self.N, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ] = arr.astype(self.dtypeNpKeras)

        self.selectorLayer.set_weights(currWeights)
    
    def setKerasSelectorBroken(self, arr, idx, out=0):
        if idx >= self.M:
            raise ValueError('Specified index must be less than the number of UO Regions!')

        currWeights = self.selectorLayer.get_weights()

        currWeights[0][:, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ] = np.zeros((self.m*self.N,self.N),dtype=self.dtypeNpKeras)
        currWeights[0][:, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ] = arr.astype(self.dtypeNpKeras)

        self.selectorLayer.set_weights(currWeights)
    
    def getKerasSelector(self, idx, out=0):
        if idx >= self.M:
            raise ValueError('Specified index must be less than the number of UO Regions!')

        currWeights = self.selectorLayer.get_weights()

        return currWeights[0][out*self.N:(out+1)*self.N, (out*(self.N*self.M)+idx*self.N):(out*(self.N*self.M)+(idx+1)*self.N) ].astype(self.dtype)
    
    def getKerasAllSelectors(self):

        return [ \
                [self.getKerasSelector(j, out=k) for j in range(self.M)] for k in range(self.m) \
            ]

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
    
    def exportONNX(self,fname=None):
        assert onnxAvailable, 'ONNX is unavailable.'
        
        if self.model is None:
            self.createKeras()

        if not self.incBias or not self.flat:
            print('ONNX export requires that createKeras() be called with options: incBias=True, flat=True.\nPlease re-create Keras model with these options...')
            return
        
        dummyBias = 1.0

        # Set a non-zero bias on the selection/minBank/maxBank layers, so that the biases survive the ONNX conversion
        wts = self.selectorLayer.get_weights()
        self.selectorLayer.set_weights([wts[0],dummyBias*np.ones(wts[1].shape,dtype=self.dtypeNpKeras)])

        for lyr in self.lays:
            if len(lyr) > 0 and len(lyr[0]) == 2:
                for l in lyr[0]:
                    wts = l.get_weights()
                    l.set_weights([wts[0], dummyBias*np.ones(wts[1].shape,dtype=self.dtypeNpKeras)])
        

        # Convert the Keras model to ONNX
        self.onnxModel, _ = tf2onnx.convert.from_keras(self.model, input_signature=(tf.TensorSpec((None, self.n), self.dtypeKeras, name="input"),), opset=13)

        # This is made available to help using the ONNX runtime to verify the output of the ONNX:
        self.onnxOutputs = [out.name for out in self.onnxModel.graph.output]

        # Change all of the selection/minBank/maxBank biases to zero in the exported ONNX:
        for i in self.onnxModel.graph.initializer:
            if re.search(r'.*(selectionLayer|minBank|maxBank)[_\d]*/BiasAdd',i.name):
                i.raw_data =  onnx.numpy_helper.from_array( np.zeros_like(onnx.numpy_helper.to_array(i)) ).raw_data


        # Reset the biases to zero in the Keras model
        wts = self.selectorLayer.get_weights()
        self.selectorLayer.set_weights([wts[0], np.zeros(wts[1].shape,dtype=self.dtypeNpKeras)])

        for lyr in self.lays:
            if len(lyr) > 0 and len(lyr[0]) == 2:
                for l in lyr[0]:
                    wts = l.get_weights()
                    l.set_weights([wts[0], np.zeros(wts[1].shape,dtype=self.dtypeNpKeras)])

        if fname is not None:
            onnx.save(self.onnxModel, fname)
    
    def save(self, fname=None):
        saveDict = {}
        saveDict['TLLFormatVersion'] = '0.1.0'
        props = ['n','N','M','m','localLinearFns','selectorSets']
        for p in props:
            saveDict[p] = getattr(self,p)
        if fname is not None:
            with open(fname,'wb') as fp:
                pickle.dump(saveDict,fp)
        return saveDict

    @classmethod
    def fromTLLFormat(cls, tllfile, dtypeKeras=tfDataType, validateFile=True):
        if type(tllfile) != dict:
            with open(tllfile, 'rb') as fp:
                tllDict = pickle.load(fp)
        else:
            tllDict = tllfile
            tllfile = 'Input dictionary'
        props = ['n','N','M','m','localLinearFns','selectorSets','TLLFormatVersion']
        if not all([p in tllDict for p in props]):
            raise(TypeError(f'{tllFile} does not contain a valid TLL Format. One or more properties are missing.'))
        if validateFile:
            props.remove('TLLFormatVersion')
            props.remove('localLinearFns')
            props.remove('selectorSets')
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
                    raise(TypeError(f'{tllfile} does not contain a valid TLL format. Element {j} of proptery \'selectorSets\' should be a list of length at least 1.'))
                for k in range(len(tllDict['selectorSets'][j])):
                    if type(tllDict['selectorSets'][j][k]) != set \
                                or any([type(el) != int for el in tllDict['selectorSets'][j][k]]) \
                                or min(tllDict['selectorSets'][j][k]) < 0 \
                                or max(tllDict['selectorSets'][j][k]) >= tllDict['N']:
                            raise(TypeError(f'{tllfile} does not contain a valid TLL format. Selector set {k} for output {j} should be a set of integers between 0 and {tllDict["N"]-1}'))
                    
        tll = cls(input_dim=tllDict['n'], output_dim=tllDict['m'], linear_fns=tllDict['N'], uo_regions=tllDict['M'], dtype=dtype, dtypeKeras=dtypeKeras)
        tll.setLocalLinearFns(tllDict['localLinearFns'])
        tll.setSelectorSets(tllDict['selectorSets'])

        return tll


    @classmethod
    def fromONNX(cls, onnxFile, dtype=npDataType):
        assert onnxAvailable, 'ONNX is unavailable.'

        importONNXModel = onnx.load(onnxFile)

        assert len(importONNXModel.graph.node) >= 8, 'ERROR: Cannot be a TLL network with fewer than eight layers.'

        assert importONNXModel.graph.node[-1].op_type == 'Add', 'ERROR: TLL layer -1 must be an Add layer.'
        assert importONNXModel.graph.node[-2].op_type == 'MatMul', 'ERROR: TLL layer -1 must be a MatMul layer.'
        assert importONNXModel.graph.node[0].op_type == 'MatMul', 'ERROR: TLL layer 0 must be a MatMul layer.'
        assert importONNXModel.graph.node[1].op_type == 'Add', 'ERROR: TLL layer 1 must be an Add layer.'
        assert importONNXModel.graph.node[2].op_type == 'MatMul', 'ERROR: TLL layer 2 must be a MatMul layer.'

        importONNXDict = createONNXDict(importONNXModel)

        m = importONNXDict[importONNXModel.graph.node[-2].name]['dims'][1]
        n = importONNXDict[importONNXModel.graph.node[0].name]['dims'][0]
        N = importONNXDict[importONNXModel.graph.node[0].name]['dims'][1]//m

        assert N*m == importONNXDict[importONNXModel.graph.node[0].name]['dims'][1], 'ERROR: TLL layer 0 must have outputs == N * m'

        M = importONNXDict[importONNXModel.graph.node[2].name]['dims'][1]//(m*N)

        assert M*m*N == importONNXDict[importONNXModel.graph.node[2].name]['dims'][1], 'ERROR: TLL layer 2 must have outputs M * m * N'

        dtypeNpKeras = onnx.numpy_helper.to_array(importONNXDict[importONNXModel.graph.node[0].name]['initializer']).dtype
        if dtypeNpKeras == np.float32:
            dtypeKeras = tf.float32
        else:
            dtypeKeras = tf.float64

        tll = cls(input_dim=n, output_dim=m, linear_fns=N, uo_regions=M, dtype=dtype, dtypeKeras=dtypeKeras)

        tll.createKeras(incBias=True,flat=True)

        tll.linearLayer.set_weights( \
            [ \
                onnx.numpy_helper.to_array(importONNXDict[importONNXModel.graph.node[0].name]['initializer']), \
                onnx.numpy_helper.to_array(importONNXDict[importONNXModel.graph.node[1].name]['initializer'])
            ] \
        )

        tll.selectorLayer.set_weights( \
            [ \
                onnx.numpy_helper.to_array(importONNXDict[importONNXModel.graph.node[2].name]['initializer']), \
                onnx.numpy_helper.to_array(importONNXDict[importONNXModel.graph.node[3].name]['initializer'])
            ] \
        )
        sSets = [[] for k in range(m)]
        for k in range(m):
            for j in range(M):
                s = np.nonzero(tll.getKerasSelector(j,out=k))
                assert set(s[1]) == set(range(N)) and np.all(tll.getKerasSelector(j,out=k)[s] == 1), 'ERROR: TLL layer 2 must contain valid selector matrices.'
                sSets[k].append(set(s[0]))
        
        tll.setLocalLinearFns(tll.getKerasAllLocalLinFns(transpose=True))
        tll.setSelectorSets(sSets)

        tll.exportONNX()

        validTLLONNXDict = createONNXDict(tll.onnxModel)

        assert len(tll.onnxModel.graph.node) == len(importONNXModel.graph.node) \
            and len(tll.onnxModel.graph.initializer) == len(importONNXModel.graph.initializer), 'ERROR: Incorrect number of layers for a TLL'
        
        for ndIdx in range(len(importONNXModel.graph.node)):
            assert importONNXModel.graph.node[ndIdx].op_type == tll.onnxModel.graph.node[ndIdx].op_type, 'ERROR: TLL layer type mismatch for layer ' + str(ndIdx)
            if importONNXModel.graph.node[ndIdx].name in importONNXDict and tll.onnxModel.graph.node[ndIdx].name in validTLLONNXDict:
                assert np.array_equal(onnx.numpy_helper.to_array( importONNXDict[importONNXModel.graph.node[ndIdx].name]['initializer'] ), \
                                        onnx.numpy_helper.to_array( validTLLONNXDict[tll.onnxModel.graph.node[ndIdx].name]['initializer'] ) ), \
                            'ERROR: TLL layer weights mismatch for layer ' + str(ndIdx)
        
        return tll



# *********************************
# *      Helper functions:        *
# *********************************

def createONNXDict(onnxModel):
    weightsDict = {}
    for inst in onnxModel.graph.initializer:
        if re.search( r'^.*/ReadVariableOp:0', inst.name ):
            name = inst.name[0:-17]
            weightsDict[name] = {'raw_data':inst.raw_data, 'dims':inst.dims, 'data_type':inst.data_type, 'initializer':inst}
    
    for ndIdx in range(len(onnxModel.graph.node)):
        if onnxModel.graph.node[ndIdx].name in weightsDict:
            weightsDict[onnxModel.graph.node[ndIdx].name]['op_type'] = onnxModel.graph.node[ndIdx].op_type
            weightsDict[onnxModel.graph.node[ndIdx].name]['node_idx'] = ndIdx

    return weightsDict 


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

def MinMaxBankByN(numGroups=1,groupSize=2,outputDim=1,maxQ=True,incBias=False,flat=False, dtypeKeras=tfDataType, layerIdx=None):

    if dtypeKeras == tf.float64:
        dtypeNpKeras = np.float64
    elif dtypeKeras == tf.float32:
        dtypeNpKeras = np.float32
    else:
        dtypeNpKeras = np.float64

    if groupSize==1:
        raise ValueError('groupSize argument must be >1 (no min/max needed otherwise!)')

    odd = np.mod(groupSize,2)==1

    if maxQ:
        finalWeights = np.array([0.5,-0.5,0.5,0.5],dtype=dtypeNpKeras)
    else:
        finalWeights = np.array([0.5,-0.5,-0.5,-0.5],dtype=dtypeNpKeras)
    

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

    compLayerKernel = np.zeros((groupSize,4*(numMins + 1) if odd else 4*numMins),dtype=dtypeNpKeras)

    for k in range(numMins):
        compLayerKernel[2*k:2*k+2, 4*k:4*(k+1)] = np.array([[1,1],[-1,-1],[-1,1],[1,-1]],dtype=dtypeNpKeras).T
    # Compares the last two elements of the block, even though one of them alread
    # appears in another comparison
    if odd:
        compLayerKernel[groupSize-2:groupSize,compLayerKernel.shape[1]-4:compLayerKernel.shape[1]] = np.array([[1,1],[-1,-1],[-1,1],[1,-1]],dtype=dtypeNpKeras).T
    
    outLayerKernel = np.zeros((compLayerKernel.shape[1], numMins + 1 if odd else numMins))

    for k in range(numMins+1 if odd else numMins):
        outLayerKernel[4*k:4*(k+1),k] = finalWeights

    # Make a block diagonal matrix; see https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
    if flat:
        compLayerKernel = np.kron(np.eye(numGroups*outputDim,dtype=int),compLayerKernel)
        outLayerKernel = np.kron(np.eye(numGroups*outputDim,dtype=int),outLayerKernel)

    nameStr = layerIdx if layerIdx is None else ('max' if maxQ else 'min') + 'Bank' + str(layerIdx)
    compLayer = Dense(compLayerKernel.shape[1],use_bias=incBias,trainable=False,activation='relu',input_shape=shapeTuple,dtype=dtypeKeras,name=(nameStr if nameStr is None else nameStr + '_0'))

    outLayer = Dense(outLayerKernel.shape[1],use_bias=incBias,trainable=False,dtype=dtypeKeras,name=(nameStr if nameStr is None else nameStr + '_1'))

    return ((compLayer,outLayer),outLayerKernel.shape[1],compLayerKernel,outLayerKernel)





# if __name__ == '__main__':
    
#     # mxnet = MinMaxBankByN(numGroups=2,groupSize=3,outputDim=2)

#     # x = TLLnet(input_dim=1, output_dim=2, linear_fns=10)

#     # t2 = TLLnet(input_dim=1, output_dim=1, linear_fns=5, uo_regions=25)

#     tFlat = TLLnet(input_dim=3,output_dim=2,linear_fns=5,uo_regions=29,incBias=True,flat=True)

#     print('done')