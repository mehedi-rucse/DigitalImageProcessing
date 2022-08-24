import matplotlib.pyplot as plt
import cv2
path='/home/mehedi/Desktop/Digital Image Processing/Labtest1/mri1.jpg'
print(path)
img=plt.imread(path)
print(img.shape)
plt.subplot(2,2,1)
plt.imshow(img)
grayscale=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
print(grayscale.shape)
plt.subplot(2,2,2)
plt.imshow(grayscale,cmap='gray')
th,binaryimage=cv2.threshold(grayscale,127,255,cv2.THRESH_BINARY)
print(binaryimage.shape)
plt.subplot(2,2,3)
plt.imshow(binaryimage,cmap='gray')
#plt.hist(img.ravel(),256,[0,255])
plt.subplot(2,2,4)
plt.hist(img.ravel(),256,[0,255])

















plt.show()



