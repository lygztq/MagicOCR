from scipy import misc
from PIL import Image
import math
import cv2
import read_file
import numpy as np

TEXT_PATH = './data/test_txt'
IMG_PATH = './data/test_image'

def generate_mask(shape,points):
	"""
	Generate the mask for a given img

	args:
		shape: the shape of the image
		points: the boundary points of true_text in the image
	"""
	mask = np.zeros(shape,dtype=np.int8)
	for pt in points:
		# change form
		pt = np.array(pt).astype(np.int32)
		pt = pt.reshape((-1,1,2))

		cv2.fillPoly(mask, [pt], (1))

	#misc.imsave('./test.png', mask)
	#print mask

	return mask

def mask_image(img, points):
	mask = generate_mask(img.shape, points)
	#img = img&mask
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if mask[i][j]==0:
				img[i][j] = 0


# if __name__ == '__main__':
# 	generate_mask([500,500],[ [[0,0],[300,0],[300,300],[0,300]] ])

