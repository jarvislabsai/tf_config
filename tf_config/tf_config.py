import os
import tensorflow as tf
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
def tf_config():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*11)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            pass
        

