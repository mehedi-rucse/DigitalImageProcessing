import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread('pic.jpg')
    print(img.shape)
    grayscale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)
    equalizedImg = cv2.equalizeHist(grayscale)
    img_set = [grayscale,equalizedImg]
    title_set = ['Grayscale Image','Equalized Image']
    plt_img(img_set,title_set)
    hist_img(img_set,title_set)
def plt_img(img_set,title_set):
    ln = len(img_set)
    plt.figure(figsize=(20,20))
    l = 1
    for i in range(ln):
        plt.subplot(2,2,l)
        plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
        l = l +2
    
def hist_img(img_set,title_set):
    ln = len(img_set)
    l = 2
    for i in range(ln):
        plt.subplot(2,2,l)
        plt.hist(img_set[i].ravel(),256,[0,256])
        plt.title(title_set[i])
        l = l +2
    plt.savefig('equalizedFig.jpg')
    plt.show()

if __name__ == '__main__':
    main()
    