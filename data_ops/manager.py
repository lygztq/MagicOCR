import os
import random

import numpy as np
from PIL import Image

class DataManager(object):
	def __init__(self, data_path, text_path, test_percentage = 0.25):
		"""
		Loads the data.

		:param data_path: The path of image files.
		:param text_path: The path to the text file containing image-text relationships.
		:param test_percentage: The percentage of test data among all data, in the range [0, 1].
		"""
		self.records = []
		self.images = []
		with open(text_path, 'r') as fin:
			for line in fin.readlines():
				split = line.strip().split('\t')
				assert len(split) >= 1
				self.records.append((split[0], '\t'.join(split[1:])))
		for filename, _ in self.records:
			img = np.array(Image.open(os.path.join(data_path, filename)))
			if len(img.shape) > 2 and img.shape[2] > 1:  # convert to gray scale; not sure if necessary
				img = img[::0]
			self.images.append(img)
		test_set_len = int(len(self.records) * test_percentage)
		self.test_set = (self.images[:test_set_len], [x[1] for x in self.records[:test_set_len]])
		self.training_set = (self.images[test_set_len:], [x[1] for x in self.records[test_set_len:]])

	# data: [batch_size, 32, max_img_width, 1], [string1, string2, ..., stringBatch_size], [length1, length2, ...,lengthBatch_size]
	def get_next_train_batch(self, batch_size = 128):
		"""
		Shorthand for get_random_batch(self.training_set, batch_size)
		"""
		return DataManager.get_random_batch(self.training_set, batch_size)

	# return three numpy array batch_x, batch_y, batch_length
	# data: [batch_size, 32, max_img_width, 1], [[string1, string2, ..., stringBatch_size], -1], [[length1, length2, ...,lengthBatch_size], -1]
	def get_next_test_batch(self, batch_size = 128):
		"""
		Shorthand for get_random_batch(self.test_set, batch_size)
		"""
		return DataManager.get_random_batch(self.test_set, batch_size)

	@staticmethod
	def get_random_batch(target_set, batch_size):
		"""
		Returns two sets of corresponding randomly selected items from the given sets.

		:param target_set: A tuple containing two sets.
		:param batch_size: The desired size of the batch.
		:return: The two sets of selected items.
		"""
		indices = random.sample(range(len(target_set[0])), batch_size)
		return np.array([target_set[0][i] for i in indices]), np.array([target_set[1][i] for i in indices])

def main():
	dm = DataManager('test/', 'test_data')
	print dm.get_next_test_batch()

if __name__ == '__main__':
	main()
