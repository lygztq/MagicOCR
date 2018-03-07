import os
import tensorflow as tf 
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from data_manager import DataManager
import config

class CRNN:
    def __init__(self, batch_size, epoches, data_path):
        self.batch_size = batch_size
        self.epoches = epoches
        self.data = DataManager(data_path, batch_size)

        self.sess = tf.Session()

        with self.sess.as_default():   
            self.inputs, self.targets, self.seq_len, self.logits, self.decoded, self.optimizer, self.accracy, self.losses,self.initlizer =  self.build_model()
            self.initlizer.run()

    def build_model(self):
        input = tf.placeholder(tf.float32, [self.batch_size, 32, -1, 1])
        target = tf.sparse_placeholder(tf.int32, [None])
        seqLength = tf.placeholder(tf.int32, [None])

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
        loss = tf.reduce_mean(tf.nn.ctc_loss(target, self.predict_out, seqLength))

        # optimizer
        optimizer = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)

        # the predict string
        predict_string, prob = tf.nn.ctc_beam_search_decoder(predict_out, seqLength)

        # accuracy for string predict
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(self.predict_string[0], tf.int32), target))

        # init 
        init = tf.global_variables_initializer()

        return input, target, seqLength, predict_out, predict_string, optimizer, acc, loss, init

    # train and save model
    def train(self):
        with self.sess.as_default():
            for i in range(self.epoches):
                iteration_loss = 0
                for batch_x, batch_y, batch_length in self.data.get_next_train_batch():  #TODO
                    data_targets = np.asarray([label_to_array(label, config.CHAR_NUMBERS) for label in batch_y])  #TODO
                    data_targets = sparse_tuple_from(data_targets)  # TODO
                    temp3, loss_val, predict_str = self.sess.run(
                        [self.optimizer, self.losses, self.decoded],
                        feed_dict = {self.inputs:batch_x, self.targets:batch_y, self.seq_len:batch_length}
                    )
                    iteration_loss += loss_val
                print "Iteration {} : loss: {}".format(i, iteration_loss)
        return None
    
    # test the trained model
    def test(self):
        pass









        


