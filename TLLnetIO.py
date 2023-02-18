from TLLnet import TLLnet as TLLnetBase

import numpy as np
import scipy.special
import scipy

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Layer
from keras import regularizers
from keras import initializers
from numpy.lib.function_base import select
import tensorflow as tf
from keras import backend as K
import keras


import math
import re
from functools import reduce
from copy import copy, deepcopy
import re
import pickle



from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import multiprocessing as mp

shared = None

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

class TLLnet(TLLnetBase):

    def __init__(self, input_dim=1, output_dim=1, linear_fns=1, uo_regions=None, dtype=npDataType, dtypeKeras=tfDataType):
        self.pool = None
        self.mgr = None
        self.returnQueue = None
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

        super().__init__(input_dim=input_dim, output_dim=output_dim, linear_fns=linear_fns, uo_regions=uo_regions)

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
                self.minMaxStartLayer = self.minMaxStartLayer-1
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

    def createOptimizedKeras(self, iterationCount=None, NUM_CPUS=4, dtypeKeras=tf.float32):
        if dtypeKeras == tf.float64:
            dtypeNpKeras = np.float64
        elif dtypeKeras == tf.float32:
            dtypeNpKeras = np.float32
        else:
            dtypeNpKeras = np.float64

        if iterationCount is None:
            iterationCount = max(self.N//2,2)
        if self.pool is None:
            if self.mgr is None:
                self.mgr = mp.Manager()
            if self.returnQueue is None:
                self.returnQueue = self.mgr.Queue()
            self.poolGlobals = { \
                    'selectorSets': deepcopy(self.selectorSets) \
                }
            self.pool = mp.Pool(NUM_CPUS, initializer=initPoolContext, initargs=(self.poolGlobals, ))

        out = 0
        subsetAssignment = [[] for s in range(len(self.selectorSets[out]))]
        subsetTree = {}

        singletons = []
        for linIdx in range(self.N):
            members = []
            for s in range(self.M):
                if linIdx in self.selectorSets[out][s]:
                    members.append(s)
            singletons.append((frozenset([linIdx]), frozenset(members) ))
        ssets = deepcopy(singletons)
        for ii in range(self.N):
            edgeWeights, nonEmptyRows, nonEmptyCols, ssetsFull = self.assembleAdjacency(ssets,singletons,dup=True if ii == 0 else False)
            ssetsNew = []
            for iterCnt in range(iterationCount):
                row_match, col_match = scipy.optimize.linear_sum_assignment(edgeWeights,maximize=True)
                goodMatchIdx = np.nonzero(edgeWeights[row_match,col_match]>=2)[0]
                if len(goodMatchIdx) == 0:
                    break
                row_match = row_match[goodMatchIdx]
                col_match = col_match[goodMatchIdx]
                for idx in range(len(row_match)):
                    addSet = ssets[row_match[idx]][0] | singletons[col_match[idx]][0]
                    ssetsNew.append((addSet, ssets[row_match[idx]][1] & singletons[col_match[idx]][1] ))
                    if ii > 0:
                        subsetTree[frozenset(addSet)] = {'c': subsetTree[ssets[row_match[idx]][0]], 'p':None, 'set':frozenset(addSet),'layer':None, 'matchWeight':edgeWeights[row_match[idx],col_match[idx]]}
                        subsetTree[ssets[row_match[idx]][0]]['p'] = subsetTree[frozenset(addSet)]
                    else:
                        subsetTree[frozenset(addSet)] = {'c':None, 'p':None, 'set':frozenset(addSet),'layer':None, 'matchWeight':edgeWeights[row_match[idx],col_match[idx]]}
                edgeWeights[row_match,col_match] = np.zeros(len(row_match),dtype=np.int32)
            ssets = ssetsNew
            for se in ssets:
                for a in se[1]:
                    subsetAssignment[a].append([se[0],len(se[1])])
            iterationCount = max(iterationCount-1,2)
            # print(f'Created ssets = {ssets}')



        for ii in range(self.M):
            subsetAssignment[ii].sort(key=(lambda x:len(x[0])),reverse=True)
        # print(f'subsetAssignment = {subsetAssignment}')
        # print(f'subset tree = {subsetTree}')


        optimizedRealizationSets = self.realizeMinTerms(out, subsetAssignment, subsetTree)
        # print(optimizedRealizationSets)

        # Begin constructing the network:
        self.optLyrs = {}
        self.optLyrs[out] = {}
        self.optLyrs['input'] = Input(shape=(self.n,), dtype=dtypeKeras)
        self.optLyrs[out]['linear'] = [ \
                    Dense(1,dtype=dtypeKeras,name='localLin_'+str(ii)+'_'+str(out)) \
                    for ii in range(self.N) \
                ]
        self.optLyrs[out]['linearOutputs'] = [ \
                    self.optLyrs[out]['linear'][ii](self.optLyrs['input'])
                    for ii in range(self.N) \
                ]
        for lyIdx in range(len(self.optLyrs[out]['linear'])):
            self.optLyrs[out]['linear'][lyIdx].set_weights( \
                                                [ \
                                                    self.localLinearFns[out][0][lyIdx,:].reshape((self.n,1)).astype(dtypeNpKeras) , \
                                                    self.localLinearFns[out][1][lyIdx].reshape((1,)).astype(dtypeNpKeras) \
                                                ] \
                                            )

        self.optLyrs[out]['output'] = self.implementRealization(out,optimizedRealizationSets, subsetTree)

        self.pool.close()
        self.pool.join()
        self.pool = None
        self.optModel = Model(self.optLyrs['input'],self.optLyrs[out]['output'])
        return

    def implementRealization(self,out,realizationSets,subsetTree):
        minRealizations = {}
        for real in realizationSets.keys():
            commons = realizationSets[real][0]
            resid = realizationSets[real][1]
            minRealizations[real] = []
            for cset in commons:
                minRealizations[real].append(self.implementCommon(out,cset,subsetTree))
            if len(resid) > 0:
                minRealizations[real].append(realizeMinMaxSet([self.optLyrs[out]['linearOutputs'][r] for r in resid],maxQ=False))
            minRealizations[real] = realizeMinMaxSet(minRealizations[real],maxQ=False)
        outLayer = realizeMinMaxSet( list(minRealizations.values()), maxQ=True )
        return outLayer

    def implementCommon(self,out,cset,subsetTree):
        assert cset in subsetTree, 'Common set not listed in tree of common sets...'
        csetObj = subsetTree[cset]
        if csetObj['layer'] is None:
            if len(csetObj['set']) == 2:
                constituents = list(csetObj['set'])
                left = self.optLyrs[out]['linearOutputs'][constituents[0]]
                right = self.optLyrs[out]['linearOutputs'][constituents[1]]
            else:
                assert csetObj['c'] is not None
                newElem = list(csetObj['set'] - csetObj['c']['set'])
                assert len(newElem) == 1
                left = self.optLyrs[out]['linearOutputs'][newElem[0]]
                right = self.implementCommon(out,csetObj['c']['set'],subsetTree)
            csetObj['layer'] = tfMinMax2(left, right, maxQ=False)
        return csetObj['layer']

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

    @classmethod
    def fromTLLFormat(cls, tllfile, dtypeKeras=tfDataType, validateFile=True):
        tllDict = cls.fromTLLFormatDict(tllfile, validateFile=validateFile)
        tll = cls(input_dim=tllDict['n'], output_dim=tllDict['m'], linear_fns=tllDict['N'], uo_regions=tllDict['M'], dtype=dtype, dtypeKeras=dtypeKeras)
        tll.setLocalLinearFns(tllDict['localLinearFns'])
        tll.setSelectorSets(tllDict['selectorSets'])

        return tll

    @classmethod
    def fromONNX(cls, onnxFile, dtype=npDataType, validateLayers=True):
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

        #######################################################################################
        # Fetch selector layer weights from the ONNX model, and convert them to selector sets #
        #######################################################################################
        selectionLayerWeights = onnx.numpy_helper.to_array(importONNXDict[importONNXModel.graph.node[2].name]['initializer'])
        # Workaround for weird behavior where selection layer biases are stripped from ONNX model
        # Probably this has to do with VNN competion re-exporting of my original ONNX models, but then needs to be investigated
        if importONNXModel.graph.node[3].name in importONNXDict:
            selectionLayerBias = onnx.numpy_helper.to_array(importONNXDict[importONNXModel.graph.node[3].name]['initializer'])
        else:
            selectionLayerBias = np.zeros(N*M,dtype=npDataType)

        sSets = [[] for k in range(m)]
        for out in range(m):
            for idx in range(M):
                # Basically code from getKerasSelector
                minTermWeights = selectionLayerWeights[out*N:(out+1)*N, (out*(N*M)+idx*N):(out*(N*M)+(idx+1)*N) ].astype(dtype)
                s = np.nonzero(minTermWeights)
                assert set(s[1]) == set(range(N)) and np.all(minTermWeights[s] == 1), 'ERROR: TLL layer 2 must contain valid selector matrices.'
                sSets[out].append(set(s[0]))

        ##################################################
        # Fetch linear layer weights from the ONNX model #
        ##################################################
        # Basically combined code from getKerasLocalLinFns and getKerasAllLocalLinFns
        currWeights = [ \
                            onnx.numpy_helper.to_array(importONNXDict[importONNXModel.graph.node[0].name]['initializer']), \
                            onnx.numpy_helper.to_array(importONNXDict[importONNXModel.graph.node[1].name]['initializer'])
                        ]
        lLinFns = [ \
                [ \
                    currWeights[0][:, (out*N):((out+1)*N) ].astype(dtype).T, \
                    currWeights[1][ (out*N):((out+1)*N) ].astype(dtype) \
                ] for out in range(m) \
            ]

        ######################################################################################
        # Set the TLL with local linear functions/selector sets obtained from the ONNX model #
        ######################################################################################
        tll.setLocalLinearFns(lLinFns)
        tll.setSelectorSets(sSets)

        ######################################################################################################
        # Validate the parameters of the other layers in the ONNX model if requested (SLOW because of Keras) #
        ######################################################################################################
        if validateLayers:
            tll.createKeras(incBias=True,flat=True)

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

    def assembleAdjacency(self, ssetsA, ssetsB, dup=False, NUM_CPUS=4):
        assert NUM_CPUS % 2 == 0, 'Please specify an even number of CPUs'

        # Divide the work of computing the adjacency matrix:
        even = True if len(ssetsB) % 2 == 0 else False
        midIdx = len(ssetsB) // 2
        groupSize = midIdx // NUM_CPUS
        rng = []
        numReturns = 0
        for idx in range(0,midIdx,groupSize):
            rng.append( [  (idx,min(idx+groupSize,midIdx)), (len(ssetsB)-min(idx+groupSize,midIdx), len(ssetsB)-idx) ] )
            numReturns += 2
        if not even:
            rng[-1].append((midIdx + 1, midIdx + 2))
            numReturns += 1

        # Create a process pool
        # returnQueue = mp.Queue()
        for ii in range(len(rng)):
            self.pool.apply_async( \
                    adjacencyWorker, \
                    ( \
                            self.selectorSets, \
                            ssetsA, \
                            ssetsB, \
                            dup, \
                            rng[ii], \
                            self.returnQueue \
                    ) \
                )

        # adjacencyWorker(self.selectorSets, ssetsA, ssetsB, dup, rng[0], returnQueue)

        # Collect the results:
        edgeWeights = np.zeros((len(ssetsA),len(ssetsB)),dtype=np.int32)
        emptyRows = set(list(range(len(ssetsA))))
        emptyCols = set([])
        repeatedSubsets = []
        while numReturns > 0:
            it = self.returnQueue.get()
            edgeWeights[:,it[0][0]:it[0][1]] = it[1]
            emptyRows = emptyRows & it[2]
            emptyCols = emptyCols | it[3]
            repeatedSubsets = repeatedSubsets + it[4]
            numReturns -= 1

        # if dup:
        #     edgeWeights = edgeWeights + edgeWeights.T

        nonEmptyRows = np.ones(len(ssetsA),dtype=np.bool8)
        nonEmptyCols = np.ones(len(ssetsB),dtype=np.bool8)
        nonEmptyRows[list(emptyRows)] = np.zeros(len(emptyRows),dtype=np.bool8)
        nonEmptyCols[list(emptyCols)] = np.zeros(len(emptyCols),dtype=np.bool8)

        # print(edgeWeights.toarray())
        # print(f'Repeated subsets = {repeatedSubsets}')
        # Assemble the worker results into one sparse adjacency matrix
        return (edgeWeights, nonEmptyRows, nonEmptyCols, repeatedSubsets)

    def realizeMinTerms(self, out, subsetAssignment, subsetTree):
        realizations = {}
        for ii in range(len(subsetAssignment)):
            r = self.pool.apply_async( \
                        realizationWorker, \
                        ( \
                            out, \
                            ii, \
                            subsetAssignment[ii], \
                            subsetTree, \
                            self.returnQueue \
                        ) \
                    )
            # r.get()
        numReceived = 0
        while numReceived < len(subsetAssignment):
            item = self.returnQueue.get()
            realizations[item[0]] = (item[1], item[2])
            numReceived += 1
        return realizations

# *********************************
# *      Helper functions:        *
# *********************************

def realizationWorker(out, idx, subsetAssignment, subsetTree, returnQueue):
    # This function takes a list of subsets that we are going to prioritze in the construction of its respective min minterm
    selectionSet = copy(shared['selectorSets'][out][idx])
    selectionList = list(selectionSet)
    selectionList.sort()
    selectionIdx = {}
    for ii in range(len(selectionList)):
        selectionIdx[selectionList[ii]] = ii
    edgeWeights = np.zeros((len(subsetAssignment),len(selectionSet)),dtype=np.int32)
    for ii in range(len(subsetAssignment)):
        setPtr = subsetTree[subsetAssignment[ii][0]]
        weight = 0
        while setPtr['c'] is not None:
            weight += setPtr['matchWeight']
            setPtr = setPtr['c']
        for jj in subsetAssignment[ii][0]:
            edgeWeights[ii,selectionIdx[jj]] = weight
    row_match, col_match = scipy.optimize.linear_sum_assignment(edgeWeights,maximize=True)
    setLookup = {}
    for ii in range(len(row_match)):
        setLookup[subsetAssignment[row_match[ii]][0]] = (ii,(row_match[ii],col_match[ii]))
    for ii in range(len(row_match)):
        setPtr = subsetTree[subsetAssignment[row_match[ii]][0]]
        while setPtr['p'] is not None:
            if setPtr['p']['set'] in setLookup:
                setLookup.pop(subsetAssignment[row_match[ii]][0])
                break
            setPtr = setPtr['p']
    validIdxs = []
    for ky in setLookup.values():
        validIdxs.append(ky[0])
    validIdxs.sort()
    row_match = row_match[validIdxs]
    col_match = col_match[validIdxs]
    fullWeights = {}
    for ii in range(len(row_match)):
        fullWeights[row_match[ii]] = edgeWeights[row_match[ii],col_match[ii]]
    row_match = np.array(sorted(row_match,key=(lambda x: fullWeights[x]),reverse=True),dtype=np.int32)
    residual = copy(selectionSet)
    final_row_match = []
    for ii in range(len(row_match)):
        if len(subsetAssignment[row_match[ii]][0] & residual) > 0:
            final_row_match.append(row_match[ii])
            residual = residual - subsetAssignment[row_match[ii]][0]
    returnQueue.put((selectionSet, [subsetAssignment[ii][0] for ii in final_row_match], residual, edgeWeights))
    return True

SUBSET = 0
IN_SELECTORS = 1
def adjacencyWorker(selectorSets, U, V, dup, rng, returnQueue):
    retVals = []
    for ivl in rng:
        retSparse = np.zeros((len(U), abs(ivl[1]-ivl[0])), dtype=np.int32)
        emptyCols = set(list(range(ivl[0],ivl[1])))
        emptyRows = set(list(range(len(U))))
        repeatedSubsets = []
        for c in range(ivl[0], ivl[1]):
            for r in ( range(len(U)) if not dup else range(c) ):
                # Compute the number of selector sets that the augmented set U[r][SUBSET] | V[c][SUBSET] belongs to
                # (only do this if they are disjoint)
                if len(U[r][SUBSET] & V[c][SUBSET]) == 0:
                    newSet = frozenset(U[r][SUBSET] | V[c][SUBSET])
                    for rIdx in range(r):
                        if U[rIdx][SUBSET] <= newSet:
                            continue
                    if max(V[c][SUBSET])<max(newSet):
                        continue
                    inSelectors = frozenset(U[r][IN_SELECTORS] & V[c][IN_SELECTORS])
                    retSparse[r, c-ivl[0]] = len(inSelectors)
                    if len(inSelectors) > 1:
                        repeatedSubsets.append(( newSet, inSelectors, (r,c), len(inSelectors)))
                    if r in emptyRows:
                        emptyRows.remove(r)
                    if c in emptyCols:
                        emptyCols.remove(c)
        #print(retSparse.toarray())
        returnQueue.put((ivl, retSparse, emptyRows, emptyCols, repeatedSubsets))
    return True

def initPoolContext(poolGlobals):
    global shared
    shared = poolGlobals

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

def selectorMatrixToSet(sMat):
    return set(list(map(lambda x: int(x), np.nonzero(sMat)[0].tolist())))


def tfMinMax2(a,b,maxQ=True,dtypeKeras=tf.float32):
    if dtypeKeras == tf.float64:
        dtypeNpKeras = np.float64
    elif dtypeKeras == tf.float32:
        dtypeNpKeras = np.float32
    else:
        dtypeNpKeras = np.float64

    l1 = Dense(4,activation='relu',use_bias=False,trainable=False,dtype=dtypeKeras)
    lOut = l1(tf.concat([a,b],1))
    l1.set_weights([np.array([[1, -1, -1, 1],[1, -1, 1, -1]],dtype=dtypeNpKeras)])

    l2 = Dense(1,activation=None,use_bias=False,trainable=False,dtype=dtypeKeras)
    lOut = l2(lOut)
    if not maxQ:
        l2.set_weights([np.array([[0.5,-0.5,-0.5,-0.5]],dtype=dtypeNpKeras).T])
    else:
        l2.set_weights([np.array([[0.5,-0.5,0.5,0.5]],dtype=dtypeNpKeras).T])

    return lOut

def realizeMinMaxSet(lyrList, maxQ=True, dtypeKeras=tf.float32):
    if dtypeKeras == tf.float64:
        dtypeNpKeras = np.float64
    elif dtypeKeras == tf.float32:
        dtypeNpKeras = np.float32
    else:
        dtypeNpKeras = np.float64
    assert len(lyrList) >= 1

    lOut = lyrList[0]
    if len(lyrList) > 1:
        for ii in range(1,len(lyrList)):
            lOut = tfMinMax2(lyrList[ii],lOut,maxQ=maxQ,dtypeKeras=dtypeKeras)

    return lOut

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



