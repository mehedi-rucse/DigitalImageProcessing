import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
	img_path = "mri1.jpg"
	print(img_path)
	rgbImg = plt.imread(img_path)
	print(rgbImg.shape)
	r,c,k=rgbImg.shape
	
	grayscale = np.zeros((r,c),dtype=np.uint8)
	grayscale = rgb2Grayscale(rgbImg)
	filteredImg = np.zeros((r,c),dtype=np.uint8)
	
	kernel1 = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
	filteredImg = convolution2d1(grayscale,kernel1)
	processed_img2 = cv2.filter2D(grayscale,-1,kernel1);

	img_set = [rgbImg,grayscale,filteredImg,processed_img2]
	title_set = ['RGB','Grayscale','DFEFf','processed_img2']
	hist_title = ["RGB Histogram","Grayscale","gjhd",'processed_img2']
	
	plt_img(img_set,title_set)
	manual_hist(img_set,hist_title,r,c)
	
def convolution2d1(image, kernel):
	m, n = kernel.shape
	c, r = image.shape
	image = np.pad(image,1,constant_values = 0)
	print(image.shape)
	new_image = np.zeros((c, r))
	for i in range(r-m):
		for j in range(c-n):
			sum = np.sum(image[i:i+m, j:j+n]*kernel)
			sum = np.rint(sum)
			sum = max(0,sum)
			sum = min(255,sum)
			new_image[i,j] = sum
	return new_image

		
def rgb2Grayscale(img):
	r,c,k = img.shape
	grayscale = np.zeros((r,c),dtype=np.uint8)
	
	red = img[:,:,0]
	green = img[:,:,1]
	blue = img[:,:,2]
	
	for i in range(r):
		for j in range(c):
			tmp = .144 * red[i,j] + .587 * green[i,j] + .299 * blue[i,j]
			if tmp > 255:
				tmp = 255
			grayscale[i,j] = tmp
			#print(grayscale[i,j])
	return grayscale
	
def plt_img(img_set,title_set):
	ch = len(img_set)
	plt.figure(figsize=(20,20))
	j=1
	for i in range(ch):
		plt.subplot(4,2,j)
		img = img_set[i]
		ln = len(img)
		if ln == 3:
			plt.imshow(img)
		else:
			plt.imshow(img,cmap = 'gray')
		plt.title(title_set[i])
		j = j + 2
	
def manual_hist(img_set,hist_title_set,r,c):
	ch = len(img_set)
	l = 2
	for i in range(ch):
		img = (img_set[i])
		histogram = np.zeros((256), dtype=int)
		for j in range(r):
			for k in range(c):
				temp = img[j,k].astype(int)
				histogram[temp] = 1 + histogram[temp]
		y =  np.arange(256)
		plt.subplot(4,2,l)
		plt.plot(y,histogram)
		plt.ylim(0,)
		plt.title(hist_title_set[i])
		l = l + 2
	plt.show()

if __name__ == '__main__':
	main()
