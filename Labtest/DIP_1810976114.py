import matplotlib.pyplot as plt
import numpy as np
import cv2
def main():
	img_path = "mri1.jpg"
	print(img_path)
	rgbImg = plt.imread(img_path)
	print(rgbImg.shape)
	
	redChannel = rgbImg[:,:,0]
	greenChannel = rgbImg[:,:,1]
	blueChannel = rgbImg[:,:,2]
	r,c,k = rgbImg.shape
	#grayscale = gray_scale(rgbImg)
	grayscale = cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
	plt.figure(figsize=(20,20))
	for k in range(8):
		bits = np.zeros((r,c),dtype=np.uint8)
		for i in range(r):
			for j in range(c):
				bits[i,j] = grayscale[i,j] & pow(2,k)
		plt.subplot(2,4,k+1)
		plt.title("Bit "+str(k)+" Slice")
		plt.imshow(bits,cmap='gray')
	#plt.savefig('slice.jpg')
	plt.show() 
	img_set = [redChannel,greenChannel,blueChannel]
	img_title = ["Red","Green","Blue",]
	hist_title = ["Red Histogram","Green Histogram","Blue Histogram" ]
	#plt_img(img_set,img_title)
	#manual_hist(img_set,hist_title,r,c)

def bit_slicing(img):
	r,c = img.shape
	
	
	
def gray_scale(rgbImg):
	r,c,k = rgbImg.shape
	red = rgbImg[:,:,0]
	green = rgbImg[:,:,1]
	blue = rgbImg[:,:,2]
	grayscale = np.zeros((r,c))
	for i in range(r):
		for j in range(c):
			grayscale[i,j]	=  0.299*red[i,j] + 0.587 * green[i,j] + 0.144 * blue[i,j]	
	return grayscale
	
def manual_hist(img_set,hist_title_set,r,c):
	ch = len(img_set)
	plt.figure(figsize=(20,20))

	for i in range(ch):
		img = img_set[i]
		histogram = np.zeros((256,), dtype=int)
		for j in range(r):
			for k in range(c):
				temp = img[j,k]
				histogram[temp] += 1
		y =  np.arange(256)

		plt.subplot(3,3,i+1)
		plt.plot(y,histogram)
		plt.ylim(0,)
		plt.fill_between(y,histogram)
		plt.title(hist_title_set[i])
	plt.show()
	
def plt_img(img_set,img_title):
	ch = len(img_set)
	plt.figure(figsize=(20,20))
	for i in range(ch):
		plt.subplot(2,2,i+1)
		img = img_set[i]
		plt.imshow(img_set[i],cmap="gray")
		plt.title(img_title[i])
	plt.show()

	
	
if __name__ == "__main__":
	main()
