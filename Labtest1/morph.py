import matplotlib.pyplot as plt
import numpy as np
import cv2
path='/home/mehedi/Desktop/Digital Image Processing/Labtest1/field.jpg'
print(path)
img=plt.imread(path)

# convert img into gray scale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# transfer gray to binary image
_,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)[1]


print(img.shape)
grayscale=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
th,binary=cv2.threshold(grayscale,127,255,cv2.THRESH_BINARY)
sturct_elm=np.ones((3,3),dtype=np.uint8)
im_erosion=cv2.erode(binary,sturct_elm,iterations=1)
plt.subplot(2,2,1)
plt.imshow(im_erosion,cmap='gray')
im_dilation=cv2.dilate(binary,sturct_elm,iterations=1)
plt.subplot(2,2,2)
plt.imshow(im_dilation,cmap='gray')
opening=cv2.morphologyEx(binary,cv2.MORPH_OPEN,sturct_elm)
plt.subplot(2,2,3)
plt.imshow(opening,cmap='gray')
closing=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,sturct_elm)
plt.subplot(2,2,4)
plt.imshow(closing,cmap='gray')







plt.show()


