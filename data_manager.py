import os
import re
import numpy as np 
import config

class DataManager(object):

     def __init__(self,data_path,batch_size):
         pass
    
    # return three numpy array batch_x, batch_y, batch_length
    # data: [batch_size, 32, max_img_width, 1], [[string1, string2, ..., stringBatch_size], -1], [[length1, length2, ...,lengthBatch_size], -1]
    def get_next_train_batch(self):
        pass 
    
    # return three numpy array batch_x, batch_y, batch_length
    # data: [batch_size, 32, max_img_width, 1], [[string1, string2, ..., stringBatch_size], -1], [[length1, length2, ...,lengthBatch_size], -1]
    def get_next_test_batch(self):
        pass