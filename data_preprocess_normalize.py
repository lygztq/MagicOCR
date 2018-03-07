import os
import math
import cv2
import read_file
import numpy as np
from PIL import Image


def transform_image(img, pts):
	"""
	Transforms the given image using the given points, so that the points are at the four corners
	of the image in the transformed image.

	:param img: The image data.
	:param pts: The four points in counter-clockwise order, and the first point denotes the desired
	            top-left corner of the transformed image.
	:return: The transformed image.
	"""
	p1, p2, p3 = np.array(pts[0]), np.array(pts[1]), np.array(pts[2])
	tgwidth, tgheight = int(math.ceil(np.linalg.norm(p3 - p2))), int(math.ceil(np.linalg.norm(p2 - p1)))
	mat = cv2.getPerspectiveTransform(
		np.float32([pts[0], pts[1], pts[3], pts[2]]),
		np.float32([[0, 0], [0, tgheight], [tgwidth, 0], [tgwidth, tgheight]])
	)
	return cv2.warpPerspective(img, mat, (tgwidth, tgheight))


def process_image(name, ext, text_folder, img_folder):
	"""
	Calls transform_image to transform all text regions in the given image, with information in the
	text file.

	:param name: The name of the image file and text file, without extension.
	:param ext: The extension of the image file.
	:param text_folder: The folder that contains the text file.
	:param img_folder: The folder that contains the image file.
	:return: A list of tuples of the format (image, text). The text is the corresponding label in the text file.
	"""
	img = np.array(Image.open(os.path.join(img_folder, name + ext)))
	points, texts = read_file.get_box_inf(text_folder, name)
	result = []
	for i in range(len(points)):
		result.append((transform_image(img, points[i]), texts[i]))
	return result


def process_folders(text_folder, image_folder, target_image_folder, target_text_file):
	"""
	Calls process_image to process all images in a folder, writes the resulting images to the given folder,
	and the resulting image-text relationships to the given text file.

	:param text_folder: The folder that contains the text files.
	:param image_folder: The folder that contains the image files.
	:param target_image_folder: The target directory to save transformed images to.
	:param target_text_file: The text file in which all image-text relationships will be stored.
	"""
	mapping = {}
	imglist = read_file.get_names_and_extensions(image_folder)
	i = 0
	for name, ext in imglist:
		i += 1
		print 'processing {} of {}'.format(i, len(imglist))
		counter = 0
		for img, text in process_image(name, ext, text_folder, image_folder):
			newname = '{}_{:03}{}'.format(name, counter, ext)
			cv2.imwrite(os.path.join(target_image_folder, newname), img)
			mapping[newname] = text
			counter += 1
	with open(target_text_file, 'w') as fout:
		print 'writing relationships'
		fout.writelines(('{}\t{}\n'.format(k, v) for k, v in mapping.items()))
	print 'done'


def main():
	try:
		os.mkdir('./test/')
	except Exception as e:
		print e.message
	process_folders('../data/txt_1000', '../data/image_1000', './test/', './test_data')


if __name__ == '__main__':
	main()
