import os
import sys


def get_names(text_dir):
	"""
	Return the list of the name of data

	args:
		text_dir: the dir that contain the txt files
	"""
	names = []
	for root, _, files in os.walk(text_dir):
		names = files
	for i in range(len(names)):
		names[i] = os.path.splitext(names[i])[0]
	return names


def get_box_inf(text_dir, name):
	"""
	Return the boundary points and true texts of one img

	args:
		text_dir: the dir that contain the txt files
		names: file_name
	"""
	path = os.path.join(text_dir, name)
	path = path + '.txt'
	with open(path, 'r') as contain:
		points = []
		texts = []

		for line in contain.readlines():
			line = line[:-1].split(',')  # remove '\n'
			true_text = ','.join(line[8:])  # the truth_ground text

			# check whether unclear image
			clear = False
			for c in true_text:
				if c != '#':
					clear = True
					break
			if not clear:
				continue

			# the boundary points
			b_index = map(float, line[:8])
			b_points = []
			for i in range(4):
				point = [b_index[2 * i], b_index[2 * i + 1]]
				b_points.append(point)
			points.append(b_points)
			texts.append(true_text)

		# return points, texts
		return points, texts

# if __name__ == '__main__':
# 	get_box_inf('./data/txt_1000','TB1.3pkLXXXXXXjaFXXunYpLFXX')
