# Machine Learning technique on PET block detector development - Part 2-5

## Crystal/Pixel discrimination for DQS PET block detector using TensorFlow (v1.0, 2019-12)   
---


```python
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline

```

    C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\framework\dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    C:\ProgramData\Anaconda3\lib\site-packages\tensorboard\compat\tensorflow_stub\dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    


```python
# re-load data

file = "D:\\ML on PET block\\new_concept_block_lso\\new_concept_block_15x15\\results\\ML_data\\new_concept_block_15x15_sorted_events1.csv"
df0 = pd.read_csv (file, comment='#')

X = df0.iloc[:,4:].values
decoding = df0.iloc[:,0:4].values

infile = open('./pickle/temp_data1','rb')
X_t, X_b, X_a, X_g, X_c, index_train, index_test = pickle.load(infile)
infile.close()

pixel_x = np.array(df0['index_x'])
pixel_y = np.array(df0['index_y'])
pixel_xy = pixel_y * 15 + pixel_x

pixel_x_train = pixel_x[index_train]
pixel_y_train = pixel_y[index_train]
pixel_xy_train = pixel_xy[index_train]

pixel_x_test = pixel_x[index_test]
pixel_y_test = pixel_y[index_test]
pixel_xy_test = pixel_xy[index_test]

X_train = X[index_train]
X_test = X[index_test]
y_train = pixel_xy[index_train]
y_test = pixel_xy[index_test]

```


```python
#from keras.utils import np_utils
#y_train1 = np_utils.to_categorical(y_train, 225)
#y_test1 = np_utils.to_categorical(y_test, 225)

```


```python
#tf.get_logger().setLevel('INFO')
#tf.get_logger().setLevel('WARNING')
tf.get_logger().setLevel('ERROR')
```


```python
feature_columns=[tf.feature_column.numeric_column('x', shape=X_train.shape[1:])]
```


```python
estimator = tf.estimator.DNNClassifier(
    feature_columns = feature_columns, 
    hidden_units = [120,30],
    n_classes = 255,
    model_dir = './train/DNN')
```


```python
# Train the estimator

# tensorflow v1.4
train_input = tf.estimator.inputs.numpy_input_fn(

# tensorflow v2.0
#train_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=200,
    shuffle=False,
    num_epochs=10)
estimator.train(input_fn = train_input, steps=None) 
```

    WARNING: Entity <bound method _DNNModel.call of <tensorflow_estimator.python.estimator.canned.dnn._DNNModel object at 0x00000237245C7608>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method _DNNModel.call of <tensorflow_estimator.python.estimator.canned.dnn._DNNModel object at 0x00000237245C7608>>: AttributeError: module 'gast' has no attribute 'Index'
    WARNING: Entity <bound method DenseFeatures.call of <tensorflow.python.feature_column.feature_column_v2.DenseFeatures object at 0x00000237246BC3C8>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method DenseFeatures.call of <tensorflow.python.feature_column.feature_column_v2.DenseFeatures object at 0x00000237246BC3C8>>: AttributeError: module 'gast' has no attribute 'Index'
    WARNING: Entity <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237246C3988>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237246C3988>>: AttributeError: module 'gast' has no attribute 'Index'
    WARNING: Entity <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237246BC9C8>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237246BC9C8>>: AttributeError: module 'gast' has no attribute 'Index'
    WARNING: Entity <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237246C3448>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237246C3448>>: AttributeError: module 'gast' has no attribute 'Index'
    




    <tensorflow_estimator.python.estimator.canned.dnn.DNNClassifier at 0x23724da4e88>




```python
#eval_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
eval_input = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test, 
    shuffle=False,
    batch_size=X_test.shape[0],
    num_epochs=1)
estimator.evaluate(eval_input,steps=None) 
```

    WARNING: Entity <bound method _DNNModel.call of <tensorflow_estimator.python.estimator.canned.dnn._DNNModel object at 0x00000237251D4488>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method _DNNModel.call of <tensorflow_estimator.python.estimator.canned.dnn._DNNModel object at 0x00000237251D4488>>: AttributeError: module 'gast' has no attribute 'Index'
    WARNING: Entity <bound method DenseFeatures.call of <tensorflow.python.feature_column.feature_column_v2.DenseFeatures object at 0x00000237249FDA48>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method DenseFeatures.call of <tensorflow.python.feature_column.feature_column_v2.DenseFeatures object at 0x00000237249FDA48>>: AttributeError: module 'gast' has no attribute 'Index'
    WARNING: Entity <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237251DAB08>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237251DAB08>>: AttributeError: module 'gast' has no attribute 'Index'
    WARNING: Entity <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237251DAC08>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237251DAC08>>: AttributeError: module 'gast' has no attribute 'Index'
    WARNING: Entity <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237249E3D48>> could not be transformed and will be executed as-is. Please report this to the AutgoGraph team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output. Cause: converting <bound method Dense.call of <tensorflow.python.layers.core.Dense object at 0x00000237249E3D48>>: AttributeError: module 'gast' has no attribute 'Index'
    




    {'accuracy': 0.7451306,
     'average_loss': 0.9465294,
     'loss': 213676.17,
     'global_step': 49665}



### Tensorflow 1.4 result
train_input = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=200,
    shuffle=False,
    num_epochs=1)
estimator.train(input_fn = train_input, steps=None) 

{'accuracy': 0.71728086,
 'average_loss': 1.2570522,
 'loss': 283775.75,
 'global_step': 4515}
 
 #### epoch=10
 {'accuracy': 0.7451306,
 'average_loss': 0.9465294,
 'loss': 213676.17,
 'global_step': 49665}
 
### Tensorflow 2.0 result
train_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=200,
    shuffle=False,
    num_epochs=10)
estimator.train(input_fn = train_input, steps=None) 
eval_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test, 
    shuffle=False,
    batch_size=X_test.shape[0],
    num_epochs=1)
estimator.evaluate(eval_input,steps=None) 

{'accuracy': 0.704284,
 'average_loss': 1.473834,
 'loss': 1.473834,
 'global_step': 55180}
 
 ## Conclusions
 * <b> The outputs of the loss from Tensorflow v1.4 and v2.0 are complete different. Obviously like the definitions of the loss are different in two versions. </b>
 * <b> The accuracy from the v1.4 is better than that from the v2.0 with the similar number of iterations, which is 0.7451306 vs 0.704284 with global_step of 49665 vs 55180. In fact, even with much less iterations (4515 global step), the accuray from v1.4 is better (0.71728086) than that from v2.0.</b>
 


```python

```
