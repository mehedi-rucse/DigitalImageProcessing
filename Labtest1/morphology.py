import cv2
import matplotlib.pyplot as plt
def main():
	path = '/home/mehedi/Desktop/Digital Image Processing/Labtest1/mri1.jpg'
	print(path)
	rgbImg = plt.imread(path)
	print(rgbImg.shape)
	plt.subplot(2,2,1)
	plt.imshow(rgbImg)
	grayscale =  cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
	print(grayscale.shape)
	plt.subplot(2,2,2)
	plt.imshow(grayscale,cmap='gray')
	
	_,binaryImg = cv2.threshold(grayscale,127,255,cv2.THRESH_BINARY)
	
	plt.subplot(2,2,3)
	plt.imshow(binaryImg,cmap='gray')
	plt.subplot(2,2,4)
	plt.hist(grayscale.ravel(),256,[0,255])
	
	
	plt.show()
	

if __name__ == '__main__':
	main()
