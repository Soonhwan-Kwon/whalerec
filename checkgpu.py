# Methods for determining if using GPU

# 1
print ("Method 1:")
import tensorflow as tf
print(tf.test.is_built_with_cuda())

# 2
print ("Method 2:")
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# 3
print ("Method 3:")
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# 4
print ("Method 4:")
from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())


