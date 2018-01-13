import tensorflow as tf
from keras.applications.vgg16 import VGG16

with tf.device('/cpu:0'):
    model = VGG16()
