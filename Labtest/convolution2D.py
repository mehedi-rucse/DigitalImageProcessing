import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
	img = plt.imread('mri1.jpg')
	print(img.shape)
	grayscale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	print(grayscale.shape)
	r,c = grayscale.shape
	process_img1 = np.zeros((r,c),dtype= np.uint8)
	process_img2 = np.zeros((r,c),dtype= np.uint8)
	kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
	process_img1 = cv2.filter2D(grayscale,-1,kernel)
	process_img2 = convolution2D(grayscale,kernel)
	
	img_set = [grayscale,process_img1,process_img2]
	title_set = ['Grayscale','Built-in Convolution','Manual Convolution']
	plt_img(img_set,title_set)
	manual_hist(img_set,title_set)
	plt_hist(img_set,title_set)
	
def convolution2D(grayscale,kernel):
	r,c = grayscale.shape
	m,n = kernel.shape
	processed_img = np.zeros((r,c),dtype= np.uint8)
	grayscale = np.pad(grayscale,1,constant_values = 0)
	print(grayscale.shape)
	
	for i in range(r):
		for j in range(c):
			x , y = (i+m),(j+n)
			if x < r and x >= 0 and y < c and y>= 0 :
				res = np.sum(grayscale[i:i+m,j:j+n]*kernel)
				res = np.rint(res)
				res = max(0,res)
				res = min(res,255)
				processed_img[i,j] = res
	return processed_img
def manual_hist(img_set,title_set):
	ln = len(img_set)
	l =  2
	for i in range(ln):
		plt.subplot(3,3,l)
		img = img_set[i]
		histogram = np.zeros((256,),dtype= int)
		r,c = img.shape
		
		for j in range(r):
			for k in range(c):
				temp = img[j,k].astype(int)
				histogram[temp]+=1
		y = np.arange(256)
		plt.plot(y,histogram)
		plt.ylim(0,)
		plt.title(title_set[i])
		l = l + 3
	
def plt_hist(img_set,title_set):
	ln = len(img_set)
	l =  3
	for i in range(ln):
		plt.subplot(3,3,l)
		img = img_set[i]
		plt.hist(img.ravel(),256,[0,256])
		plt.title(title_set[i])
		l = l + 3
		
	plt.show()

	

def plt_img(img_set,title_set):
	ln = len(img_set)
	plt.figure(figsize=(20,20))
	l = 1
	for i in range(ln):
		plt.subplot(3,3,l)
		plt.imshow(img_set[i],cmap='gray')
		plt.title(title_set[i])
		l = l + 3

	
	
if __name__ == '__main__':
	main()
