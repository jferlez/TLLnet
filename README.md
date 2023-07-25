# TLLnet

The `TLLnet` Python package implements basic functionality for Two-Level Lattice (TLL) Neural Networks (NNs). This functionaility includes:

* Saving/Loading TLL NNs to disk
* Evaluating TLL NNs on inputs
* Converting TLL NNs to Keras/ONNX models

For a primer on TLL NNs, please see:

>_AReN: Assured ReLU NN Architecture for Model Predictive Control of LTI Systems._  
>James Ferlez and Yasser Shoukry. HSCC '20: 23rd ACM International Conference on Hybrid Systems: Computation and Control, May 2020. Article No.: 6. Pages 1–11. [https://doi.org/10.1145/3501710.3519533](https://dl.acm.org/doi/10.1145/3365365.3382213)

_Please contact [jferlez@uci.edu](mailto:jferlez@uci.edu) with any questions/bug reports._

## 1) Prerequisites

This package implements two classes: `TLLnet` and `TLLnetIO`. These classes have purposely different prerequisites:

`TLLnet` Prerequisites:
* [Numpy](https://numpy.org/) (Python)
* [Scipy](https://scipy.org/) (Python)
* _Optional for fast TLL evaluation:_ [Numba](https://numba.pydata.org/) (Python)

`TLLnetIO` Prerequisites:
* `TLLnet` prerequisites listed above
* [TensorFlow/Keras](https://keras.io/) (Python)
* _Optional for ONNX output:_ [ONNX](https://github.com/onnx/onnx) (Python)


## 2) Basic Usage

### Constructor

To create a `TLLnet` instance, call its constructor using the following keyword arguments (with example values):

```Python
tllInst = TLLnet(input_dim=2, output_dim=1, linear_fns=10, uo_regions=10)
```
These keyword arguments have the following meanings/default values:
<dl>
	<dt><tt>'input_dim'</tt></dt>
    <dd>the input dimension of the TLL, aka <tt>n</tt> (Integer; default = 1)</dd>
	<dt><tt>'output_dim'</tt></dt>
    <dd>the output dimension of the TLL, aka <tt>m</tt> (Integer; default = 1)</dd>
    <dt><tt>'linear_fns'</tt></dt>
    <dd>the number of local linear functions of the TLL, aka <tt>N</tt> (Integer; default = 1)</dd>
    <dt><tt>'uo_regions'</tt></dt>
    <dd>the number of selector sets/UO regions of the TLL, aka <tt>M</tt> (Integer; default = <tt>sum([scipy.special.binom((N**2-N)/2,i) for i in range(N+1)])</tt>)</dd>
</dl>

> **NOTE:** `TLLnet` instances are instantiated with empty local linear functions and selector sets. These must be subsequently set as described below.

### Setting Local Linear Functions

To specify the local linear functions of a `TLLnet` instance `tllInst`, use the `setLocalLinearFns` method. That is, 
```Python
tllInst.setLocalLinearFns(localLinearFns)
```
where the single argument `localLinearFns` is as follows:
<dl>
	<dt><tt>localLinearFns</tt></dt>
    <dd>A Python list of length equal to <tt>output_dim</tt> (aka <tt>m</tt>), each element of which is a Python list of length 2; each of these sub-lists specified the weights and biases for that output of the local linear functions for that TLL output, specified as Numpy arrays.
</dl>

Using the parameters in the example above, the following is a valid call to `tllInst.setLocalLinearFns`:
```Python
import numpy as np
tllInst.setLocalLinearFns([
    [np.ones((10, 2)), np.zeros((10,1))]
])
```
since that TLL has 2 inputs, 1 output and 10 local linear functions.

### Setting Selector Sets

To specify the selector sets of a `TLLnet` instance `tllInst`, use the `setSelectorSets` method. That is, 
```Python
tllInst.setSelectorSets(selectorSets)
```
where the single argument `selectorSets` is as follows:
<dl>
	<dt><tt>selectorSets</tt></dt>
    <dd>A Python list of length equal to <tt>output_dim</tt> (aka <tt>m</tt>), each element of which is a Python list of length at most <tt>M</tt>; each of these sub-lists is a list of Python sets, which are individually subsets of <tt>set(i for i in range(N))</tt>.
</dl>

Using the parameters in the example above, the following is a valid call to `tllInst.setLocalLinearFns`:
```Python
import numpy as np
tllInst.setSelectorSets([
    [{0}, {0,1}, {0,1,2}, {0,1,2,3}, {0,1,2,3,4}, {0,1,2,3,4,5}, {5,6}, {6,7}, {8,9}, {9}]
])
```
since that TLL has 1 output, 10 local linear functions and 10 selector sets.


## 3) Saving TLL NNs to Disk
The `TLLnet` class provides a `save` method that can be used to save a TLL to disk. This method has two modes of operation.

### Invoking `save` with no Arguments

When invoked with no arguments, the `save` method returns an ordinary Python containing all of the information needed to reconstruct the TLL. For example:
```Python
tllDict = tllInst.save()
```
will produce a Python dictionary `tllDict` which has keys:
<dl>
	<dt><tt>'n'</tt></dt>
    <dd>the input dimension of the TLL</dd>
	<dt><tt>'m'</tt></dt>
    <dd>the output dimension of the TLL</dd>
    <dt><tt>'N'</tt></dt>
    <dd>the number of local linear functions of the TLL</dd>
    <dt><tt>'M'</tt></dt>
    <dd>the number of selector sets/UO regions of the TLL</dd>
    <dt><tt>'localLinearFuns'</tt></dt>
    <dd>a list containing the local linear functions for each TLL output in the form:  [W, b] where W is an (N x n) dimensional Numpy array and b is an (N x 1) dimensional Numpy array</dd>
    <dt><tt>'selectorSets'</tt></dt>
    <dd>a list containing the selector sets for each TLL output as a list of Python <tt>frozensets</tt>, with each such frozenset a subset of <tt>set(i for i in range(N))</tt></dd>
    <dt><tt>'TLLFormatVersion'</tt></dt>
    <dd>a string specifying the format version number (currently <tt>'0.1.0'</tt>)</dd>
</dl>

Dictionaries in this format can be used to create a new TLL instance by supplying them as an argument to the class method `TLLnet.fromTLLFormat`. That is, the following code effectively "deep copies" `tllInst` into `tllInst2`:
```Python
tllDict = tllInst.save()
tllInst2 = TLLnet.fromTLLFormat(tllDict)
```
> **NOTE:** This allows TLL instances to be passed between Python processes with minimal `pickle` overhead, since only Python built-in objects and buffer-based objects are contained in these dictionaries. It also allows users to easily serialize TLL objects using a serialization protocol of their choice.

### Invoking `save` with One Argument

The `save` method of `TLLnet` takes an optional keyword argument `fname=`; this argument accepts a string containing a file name, and saves the TLL to disk in a file with that name. For example:
```Python
tllInst.save(fname='my_tll.tll')
```
will save the TLL to the file `my_tll.tll`.

TLLs saved in this way can be loaded using the `TLLnet.fromTLLFormat` class method by simply providing a file name string as an argument (instead of a dictionary as in [Invoking `save` with no Arguments](#invoking-save-with-no-arguments)). For example, the following code will load the file created above into a new TLL instance:
```Python
tllInst2 = TLLnet.fromTLLFormat('my_tll.tll')
```

> **NOTE:** Internally, this usage creates a Python dictionary as in [Invoking `save` with no Arguments](#invoking-save-with-no-arguments), and saves it to disk using `pickle.dump`. Thus, these files can be manuallly loaded and examined using code such as:
> ```Python
> import pickle
> with open('my_tll.tll','rb') as fp:
>   tllDict = pickle.load(fp)
> ```
> with `tllDict` containing keys/data as described above in [Invoking `save` with no Arguments](#invoking-save-with-no-arguments) .

> **NOTE:** It is expected that future versions of `TLLnet` will implement saving with the option of selecting other serialization protocols.

## 4) Working with TLL NNs in other NN Formats

To convert TLLs to other formats, you should work with an instance of the `TLLnetIO` class, which subclasses the `TLLnet` class. To obtain an instance of the former from an instance of the latter, you can use the export features described above:
```Python
tllIOInst = TLLnet.TLLnetIO.fromTLLFormat(tllInst.save())
```
The call `tllInst.save()` creates a Python dictionary description of the `TLLnet` instance `tllInst`; the call `TLLnet.TLLnetIO.fromTLLFormat` then loads that dictionary into a new instance of the `TLLnetIO` class. See also Section [3) Saving TLL NNs to Disk](#3-saving-tll-nns-to-disk).

Instances of `TLLnetIO` have several additional methods for manipulating and exporting to Keras and ONNX formats (if ONNX is available).

### On NN Framework Datatypes

The constructor for `TLLnetIO` has an additional keyword parameter `dtypeKeras=` to specify the data type to use for Keras models; the class method `fromTLLFormat` responds to the same keyword argument. For example:

```Python
import tensorflow as tf

tllIOInst = TLLnet.TLLnetIO(input_dim=2, output_dim=1, linear_fns=10, uo_regions=10, dtypeKeras=tf.float32)
tllIOInst = TLLnet.TLLnetIO.fromTLLFormat(tllInst.save(),dtypeKeras=tf.float64)

```

### Working TLL NNs as Keras Models

To create a Keras model of a `TLLnetIO` instance, use the `createKeras` method:
```Python
tllIOInst.createKeras()
```
The result will be a new property of `tllIOInst` called `model`; this property contains a compiled Keras model implementation of the TLL NN.

> **NOTE:** Subsequent calls to `tllIOInst.setLocalLinearFns` or `tllIOInst.setSelectorSets` will automatically update the Keras model stored in the `model` property.

The `model` property can be used just as any other Keras model, including for export and prediction:

```Python
tllIOInst.model.save('my_tll.keras')

# Assuming tllIOInst.n == 2
import numpy as np
outpus = tllIOInst.model.predict(np.random.random((1000, 2)))
```

### Exporting TLL NNs to ONNX

To export an ONNX model of a TLL NN, first (re-)create a Keras model using the `createKeras` method with the following arguments:
```Python
tllIOInst.createKeras(incBias=True,flat=True)
```
Now it is possible to export an ONNX file using the `exportONNX` method:
```Python
tllIOInst.exportONNX(fname='my_tll.onnx')
```
which will save an ONNX implementation of `tllIOInst` in the file `my_tll.onnx`.