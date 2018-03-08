import os
import sys
import tensorflow as tf
import argparse

from crnn import CRNN
from data_ops.manager import DataManager
import config
from utils import label_to_array, levenshtein, sparse_tuple_from, ground_truth_to_word

MODEL_HYPER = None

def start_train():
	model = CRNN(MODEL_HYPER.batch_size, MODEL_HYPER.epoches, MODEL_HYPER.data_path)
	model.train()
	model.save()

def train(_):
	# preprocessing something here before training
	if tf.gfile.Exists(MODEL_HYPER.log_path):
		tf.gfile.DeleteRecursively(MODEL_HYPER.log_path)
	tf.gfile.MakeDirs(MODEL_HYPER.log_path)

	start_train()

if __name__ == '__main__':
	# parse the model hyperparameters
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--batch_size",
		type=int,
		default=config.DEFAULT_BATCH_SIZE,
		help="the size of data batch feed into the model."
	)

	parser.add_argument(
		"--epoches",
		type=int,
		default=config.DEFAULT_EPOCHES,
		help="the number of epoch in training."
	)

	parser.add_argument(
		"--data_path",
		type=str,
		default=config.DEFAULT_DATA_PATH,
		help="the path that you store data for models."
	)

	parser.add_argument(
		"--log_path",
		type=str,
		default=config.DEFAULT_LOG_PATH,
		help="the path that you store logs."
	)

	parser.add_argument(
		"--model_path",
		type=str,
		default=config.DEFAULT_MODEL_PATH,
		help="the path that you store your models."
	)
	MODEL_HYPER = parser.parse_known_args()
	tf.app.run(main=train, argv=[sys.argv[0]])
