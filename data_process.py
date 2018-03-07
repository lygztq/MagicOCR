# coding=utf-8
import sys
import os
import numpy as np
from scipy import misc
from scipy.ndimage import filters
from PIL import Image
import math
import cv2
import cPickle
import mask
import read_file

IMAGE_DIR = "./data/test_image"
TEXT_DIR = "./data/test_txt"

# IMAGE_DIR = "./data/image_1000"
# TEXT_DIR = "./data/txt_1000"


def rgb2grey(img):
	grey_img = np.zeros(img.shape[:-1],dtype=int)
	grey_img = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
	return grey_img

def pow_trans(img):
	# fix the image after resize
	max_elem = np.max(img)
	min_elem = np.min(img)
	diff = max_elem - min_elem
	img = (img-min_elem)*1.0/diff*255
	return img

def binary_img(img):
	total = 0
	cnt = int(img.shape[0]) * int(img.shape[1])
	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			total += int(img[i,j])
			# ave = ave + (img[i,j]-ave)/(cnt+1)
			# cnt+=1
	ave = total/cnt

	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			if img[i,j] >= ave:
				img[i,j] = 255
			else:
				img[i,j] = 0

def calculate_line(x1,y1,x2,y2):
	k = (y2-y1)/(x2-x1)
	b = y1 - k*x1
	return [k,b]

def calculate_y(x,k,b):
	return k*x + b

def distance(x1,y1,x2,y2):
	return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))


def main():
	names = read_file.get_names(TEXT_DIR)
	total_num = len(names)
	cnt = 0

	true_txt_list = []
	img_path_list = []

	for name in names:
		cnt+=1
		print "processing: %d / %d" % (cnt,total_num)

		# data informations
		img_path = os.path.join(IMAGE_DIR, name+'.jpg')
		points, texts = read_file.get_box_inf(TEXT_DIR, name)

		# open img
		img = misc.imread(img_path)
		img = rgb2grey(img)
		# print points

		# mask the image
		mask.mask_image(img, points)

		for i in range(len(points)):
			max_x = 0
			min_x = 100000
			max_y = 0
			min_y = 100000
			for pt in points[i]:
				# find the boundary of subimage
				max_x = max(pt[0],max_x)
				max_y = max(pt[1],max_y)
				min_x = min(pt[0],min_x)
				min_y = min(pt[1],min_y)
			
			# avoid empty area
			min_x = max(min_x-5,0)
			min_y = max(min_y-5,0)

			sub_image = img[int(min_y):int(max_y), int(min_x):int(max_x)]
			sub_image = misc.imresize(sub_image,2.0)
			sub_image_name = name + ('_%03d' % i) + '.jpg'
			save_path = os.path.join("./data/slices", sub_image_name)

			img_path_list.append(save_path)
			true_txt_list.append(texts[i])

			# save image
			print save_path
			misc.imsave(save_path, sub_image)

	cPickle.dump(img_path_list, open("./data/img_path_list.bin","wb"))
	cPickle.dump(true_txt_list, open("./data/true_text_list.bin","wb"))
	# new_image = misc.imrotate(new_image, angle=(math.atan(k)/math.pi)*180)
if __name__ == '__main__':
	main()