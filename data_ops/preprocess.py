import os
import cv2

import numpy as np
from PIL import Image
from scipy import misc

import loader
import utils

RESIZE_WIDTH = 32

def rgb2grey(img):
	grey_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
	return resize(grey_img)

def resize(img, width=RESIZE_WIDTH):
	ratio = width*1.0/img.shape[0]
	new_size = (width, max(int(ratio*img.shape[1]), 1))
	#print ratio
	img = pow_trans(misc.imresize(img, ratio, interp='bicubic'))
	return img

def pow_trans(img):
	# fix the image after resize
	max_elem = np.max(img)
	min_elem = np.min(img)
	diff = max_elem - min_elem
	if diff==0: return img
	img = (img - min_elem) * 1.0 / diff * 255
	return img

def binary_img(img):
	total = 0
	cnt = int(img.shape[0]) * int(img.shape[1])
	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			total += int(img[i, j])
		# ave = ave + (img[i,j]-ave)/(cnt+1)
		# cnt+=1
	ave = total / cnt

	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			if img[i, j] >= ave:
				img[i, j] = 255
			else:
				img[i, j] = 0

IMAGE_PROCESS_METHOD = rgb2grey

def transform_image(img, pts):
	"""
	Transforms the given image using the given points, so that the points are at the four corners
	of the image in the transformed image.

	:param img: The image data.
	:param pts: The four points. Preferably, the points should be in counter-clockwise order,
	            and the first point should denote the desired top-left corner of the transformed image.
	:return: The transformed image.
	"""
	TANGENT_UPDOWN = 1
	TANGENT_SIDE = 0.5

	# try to correct clockwise or upside-down rectangles
	p1, p2, p3 = np.array(pts[0]), np.array(pts[1]), np.array(pts[2])
	d12, d23 = p2 - p1, p3 - p2
	if utils.det(d12, d23) > 0.0:  # clockwise
		pts[1], pts[3] = pts[3], pts[1]
		p2 = np.array(pts[1])
		d12, d23 = p2 - p1, p3 - p2
	heightv, widthv = np.linalg.norm(d12), np.linalg.norm(d23)
	if d12[1] < 0 and d12[1] * TANGENT_UPDOWN > -abs(d12[0]):  # upside-down
		pts = pts[2:] + pts[:2]
	elif abs(d12[0]) * TANGENT_SIDE > d12[1] and widthv < heightv:  # side
		if d12[0] < 0:
			pts = pts[1:] + pts[:1]
		else:
			pts = pts[3:] + pts[:3]
		widthv, heightv = heightv, widthv
	# transform the image
	tgwidth, tgheight = int(widthv), int(heightv)
	mat = cv2.getPerspectiveTransform(
		np.float32([pts[0], pts[1], pts[3], pts[2]]),
		np.float32([[0, 0], [0, tgheight], [tgwidth, 0], [tgwidth, tgheight]])
	)  # the transformation matrix
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
	points, texts = loader.load_text_regions(text_folder, name)
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
	imglist = loader.get_names_and_extensions(image_folder)
	i = 0
	for name, ext in imglist:
		i += 1
		print 'processing {} of {}'.format(i, len(imglist))
		counter = 0
		for img, text in process_image(name, ext, text_folder, image_folder):
			try:
				newname_origin = '{}_{:03}_1{}'.format(name, counter, ext)
				newname_rotate = '{}_{:03}_2{}'.format(name, counter, ext)
				rotate_img = np.transpose(img,(1,0,2))

				if len(img.shape) == 3 and img.shape[-1] >= 3:  # not gray scale, need conversion
					img = IMAGE_PROCESS_METHOD(img)
					rotate_img = IMAGE_PROCESS_METHOD(rotate_img)
				flag1 = True
				flag2 = True
				if img.shape[0] < RESIZE_WIDTH/2 or img.shape[1] < RESIZE_WIDTH/2: flag1 = False
				if rotate_img.shape[0] < RESIZE_WIDTH or rotate_img.shape[1] < RESIZE_WIDTH: flag2 = False

				if flag1:
					cv2.imwrite(os.path.join(target_image_folder, newname_origin), img)
					mapping[newname_origin] = text

				if flag2:
					cv2.imwrite(os.path.join(target_image_folder, newname_rotate), rotate_img)
					mapping[newname_rotate] = text
				counter += 1
			except Exception as e:
				print "Error when processing: {}_{}".format(name, counter), "\terror: ", e.message 
	with open(target_text_file, 'w') as fout:
		print 'writing relationships'
		fout.writelines(('{}\t{}\n'.format(k, v) for k, v in mapping.items()))
	print 'done'

def main():
	try:
		os.mkdir('./test/')
	except Exception as e:
		print e.message
	if not os.path.exists('../data/slices'):
		os.mkdir('../data/slices')
	process_folders('../data/test_txt', '../data/test_image', '../data/slices', '../data/relationship')
	#process_folders('../data/txt_1000', '../data/image_1000', '../data/slices', '../data/relationship')
	#process_folders('../../data/txt_1000', '../../data/image_1000', './test/', './test_data')

if __name__ == '__main__':
	main()
