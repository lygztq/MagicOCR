from MagicOCR import config
import numpy as np

def label_to_array(label, letters):
	return [letters.index(x) for x in label]

def sparse_tuple_from(sequences, dtype = np.int32):
	indices = []
	values = []

	for n, seq in enumerate(sequences):
		indices.extend(zip([n] * len(seq), [i for i in range(len(seq))]))
		values.extend(seq)

	indices = np.asarray(indices, dtype = np.int64)
	values = np.asarray(values, dtype = dtype)
	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype = np.int64)

	return indices, values, shape

def levenshtein(s1, s2):
	if len(s1) < len(s2):
		return levenshtein(s2, s1)

	# len(s1) >= len(s2)
	if len(s2) == 0:
		return len(s1)

	previous_row = range(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1
			deletions = current_row[j] + 1
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]

def ground_truth_to_word(ground_truth):
	return ''.join([config.CHAR_DICTIONARY[i] for i in ground_truth])

def det(v1, v2):
	"""
	Calculates the cross product of two 2D vectors.

	:param v1: The left hand side operand.
	:param v2: The right hand side operand.
	:return: The cross product x1 * y2 - x2 * y1.
	"""
	return v1[0] * v2[1] - v1[1] * v2[0]
