# coding=utf-8
import sys
import os
import numpy as np
from scipy import misc
from scipy.ndimage import filters
from PIL import Image
import cPickle

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


def main():
	# build the result dir
	if not os.path.exists("./data/slices"):
		os.mkdir("./data/slices")

	# create the image path list and the true text list
	img_path_list = []
	true_txt_list = []

	# first get the file name list in the TEXT_DIR
	for root, _, files in os.walk(TEXT_DIR):
		total_num = len(files)
		total_cnt = 0
		for f in files:
			try:
				pic_path = os.path.join(IMAGE_DIR, f[:-3]+'jpg')
				file_path = os.path.join(TEXT_DIR, f)

				img = misc.imread(pic_path,mode='RGB')
				img = rgb2grey(img)
				file_contains = open(file_path)

				cnt = 0
				for i in file_contains:
					i = i.split(',')
					contain = i[8:] # the truth_ground text
					contain = ','.join(contain)
					#print contain

					contain = contain[:-1]

					# check whether the true txt is unclear(we don't use unclear subimage)
					clear = False
					for c in contain:
						if c != '#':
							clear = True
					if not clear:
						continue

					# the boundary of the subimage
					bbox = i[:8]
					bbox = map(float,bbox) # type convert

					x_boundary = [int(bbox[0]+0.5), int(bbox[0]+0.5)]
					y_boundary = [int(bbox[1]+0.5), int(bbox[1]+0.5)]

					for b in range(0,8,2):
						x_boundary[0] = min(x_boundary[0], int(bbox[b]+0.5))
						x_boundary[1] = max(x_boundary[1], int(bbox[b]+0.5))

					for b in range(1,8,2):
						y_boundary[0] = min(y_boundary[0], int(bbox[b]+0.5))
						y_boundary[1] = max(y_boundary[1], int(bbox[b]+0.5))

					y_boundary[0] = max(y_boundary[0]-5, 0)
					x_boundary[0] = max(x_boundary[0]-5, 0)

					new_image = img[y_boundary[0]:y_boundary[1],x_boundary[0]:x_boundary[1]]
					new_image_name = f[:-4] + ('_%03d' % cnt)

					new_image = misc.imresize(new_image, 2.0)
					#binary_img(new_image)
					new_image = pow_trans(new_image)

					new_image_path = os.path.join("./data/slices", new_image_name)
					new_image_path = new_image_path+'.jpg'
					
					img_path_list.append(new_image_path)
					true_txt_list.append(contain)

					misc.imsave(new_image_path, new_image)
					cnt+=1
				total_cnt+=1
				print "processing: %d / %d" % (total_cnt,total_num)
			except:
				print "get error in", f
	cPickle.dump(img_path_list, open("./data/img_path_list.bin","wb"))
	cPickle.dump(true_txt_list, open("./data/true_text_list.bin","wb"))

if __name__ == '__main__':
	main()