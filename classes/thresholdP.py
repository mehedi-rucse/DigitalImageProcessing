import cv2
import numpy as np
img_path = '/home/mehedi/Desktop/Digital Image Processing/Pic/rad.png'
img = cv2.imread(img_path,0)

#Binary threshold 0/1
_,th1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
#Inverse of Binary threshold
_,th2 = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
#Trunc->Unchange upto 126 and remain pixels are 127
_,th3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
'''TOZERO->Remain Zero Until 126 and Remain Unchanged of orginal
    Image after 127 '''
_,th4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#Opposite of Threshold_TOZERO
_,th5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)



cv2.imshow('Th4',th4)
cv2.imshow('Th5',th5)
cv2.imshow('Image',img)
cv2.imshow('Th',th1)
cv2.imshow('Th2',th2)
cv2.imshow('Th3',th3)


cv2.waitKey(0)
cv2.destroyAllWindows()