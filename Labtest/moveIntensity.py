import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
	path = '/home/mehedi/Desktop/Digital Image Processing/Labtest/mri1.jpg'
	print(path)
	rgbImg = plt.imread(path)
	print(rgbImg.shape)
	grayscale = cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
	print(grayscale.shape)
	r,c = grayscale.shape
	process_img = np.zeros((r,c),dtype = np.uint8)
	process_img1 = np.zeros((r,c),dtype = np.uint8)
	process_img2 = np.zeros((r,c),dtype = np.uint8)
	
	process_img = move_intensity(grayscale.copy(),150)
	
	
	kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
	process_img1 = convolution2D(grayscale,kernel)
	process_img2 = cv2.filter2D(grayscale,-1,kernel)
	
	img_set = [grayscale,process_img,process_img1,process_img2]
	title_set = ['Grayscale','Processed Img','Processed Img1','Processed Img2']
	
	plt_img(img_set,title_set)
	img_hist(img_set,title_set)
	
	
def convolution2D(img,kernel):
	r,c = img.shape
	m,n = kernel.shape
	img = np.pad(img,1,constant_values = 0)
	process_img = np.zeros((r,c),dtype= np.uint8)
	for i in range(r):
		for j in range(c):
			cx,cy = (i+m),(j+n)
			if cx < r and cx >=0 and cy <c and cy >= 0 :
				res = np.sum(img[i:i+m,j:j+n]*kernel)
				res = min(255,res)
				res = max(0,res)
				res = np.rint(res)
				process_img[i,j] = res
	return process_img
	
def plt_img(img_set,title_set):
	ln = len(img_set)
	plt.figure(figsize=(20,20))
	l = 1
	for i in range(ln):
		plt.subplot(4,2,l)
		img = img_set[i]
		ch = len(img)
		if ch == 3:
			plt.imshow(img)
		else :
			plt.imshow(img,cmap='gray')
		plt.title(title_set[i])
		l=l+2

def img_hist(img_set,title_set):
	ln = len(img_set)
	l = 2
	for i in range(ln):
		plt.subplot(4,2,l)
		img = img_set[i]
		histogram = np.zeros((256,),dtype = int)
		r,c = img.shape
		for j in range(r):
			for k in range(c):
				temp = img[j,k]
				histogram[temp]+=1
		y = np.arange(256)
		plt.plot(y,histogram)
		plt.ylim(0,)
		plt.title(title_set[i])
		l=l+2
	plt.show()
	
def move_intensity(img,d):
	r,c = img.shape
	for j in range(r):
		for k in range(c):
			temp = img[j,k] + d	
			temp = min(255,temp)
			temp = max(0,temp)
			img[j,k] = temp
	return img
				
	
	

if __name__ == '__main__':
	main()
