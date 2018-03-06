import os
import tensorflow as tf 
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import config

class CRNN:
    def __init__(self, batch_size, data, label):
        self.batch_size = batch_size
        self.data = data
        self.label = label

        self.input = tf.placeholder(tf.float32, [self.batch_size, 32, -1, 1])
        self.target = tf.sparse_placeholder(tf.int32, [None])
        self.seqLength = tf.placeholder(tf.int32, [None])

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
        rnn_outputs, temp1, temp2 = rnn.static_bidirectional_rnn(forward_cell, backward_cell, sequlize_2, dtype=tf.float32)

        logits = tf.reshape(rnn_outputs, [-1, 512])
        weight = tf.Variable(tf.truncated_normal([512, config.CHAR_NUMBERS], stddev=0.1), name="weight")
        bias = tf.Variable(tf.constant(0., shape=[config.CHAR_NUMBERS]), name="bias")

        logits = tf.matmul(logits, weight) + bias  # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        logits = tf.reshape(logits, [self.batch_size, -1, config.CHAR_NUMBERS])  # [batch_size, 1, CHAR_NUMBERS]

        # outputs    [[[1], [1], ... batch_size number ..., [1]], ... CHAR_NUMBER ... ]   '1' stands predict char with dtype=tf.int32
        predict_out = tf.transpose(logits, (1, 0, 2))  #  Permutes the dimensions according to perm(axis location arrange)

        # loss value to be optimized
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.target, self.predict_out, self.seqLength))

        # the predict string
        self.predict_string, prob = tf.nn.ctc_beam_search_decoder(predict_out, self.seqLength)

        # accuracy for string predict
        self.acc = tf.reduce_mean(tf.edit_distance(tf.cast(self.predict_string[0], tf.int32), self.targets))

    # train and save model
    def train(self):
        pass
    
    # test the trained model
    def test(self):
        pass        









        


