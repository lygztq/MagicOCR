import os
import tensorflow as tf 
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

class CRNN:
    def __init__(self, batch_size, data, label):
        self.batch_size = batch_size
        self.data = data
        self.label = label

        self.input = tf.placeholder(tf.float32, [self.batch_size, 32, -1, 1])
        self.build_model(self.input)
    
    def build_model(self, input):
        # CNN layers
        conv_layer1 = layers.conv2d(inputs=input, 64, kernel_size=(3,3), strides=1, padding="SAME")
        pool_layer1 = layers.max_pooling2d(inputs=conv_layer1, pool_size=(2, 2), strides=2)

        conv_layer2 = layers.conv2d(inputs=pool_layer1, 128, kernel_size=(3,3), strides=1, padding="SAME")
        pool_layer2 = layers.max_pooling2d(inputs=conv_layer2, pool_size=(2, 2), strides=2)

        conv_layer3 = layers.conv2d(inputs=pool_layer2, 256, kernel_size=(3, 3), strides=1, padding="SAME")
        conv_layer4 = layers.conv2d(inputs=conv_layer3, 256, kernel_size=(3, 3), strides=1, padding="SAME")
        pool_layer3 = layers.max_pooling2d(inputs=conv_layer4, pool_size=(1,2), strides=2)

        conv_layer5 = layers.conv2d(inputs=pool_layer3, 512, kernel_size=(3, 3), strides=1, padding="SAME")
        batch_normalize1 = layers.batch_normalization(conv_layer5)

        conv_layer6 = layers.conv2d(inputs=batch_normalize1, 512, kernel_size=(3,3), strides=1, padding="SAME")
        batch_normalize2 = layers.batch_normalization(conv_layer6)
        pool_layer4 = layers.max_pooling2d(inputs=batch_normalize2, pool_size=(1,2), strides=2)

        conv_layer7 = layers.conv2d(inputs=pool_layer4, 512, kernel_size=(2,2), strides=1, padding="VALID")

        # Map to sequence
        sequlize_1 = tf.squeeze(conv_layer7, [1])  # Removes dimensions of size 1 from the shape of a tensor
        sequlize_2 = tf.unstack(sequlize_1) # unstack one dimesion to get sequencical data

        # BiLSTM
        forward_cell = rnn.BasicLSTMCell(256)  # Initialize the basic LSTM cell
        backward_cell = rnn.BasicLSTMCell(256)  
        self.outputs, temp1, temp2 = rnn.static_bidirectional_rnn(forward_cell, backward_cell, sequlize_2, dtype=tf.float32)
    
    




        


