import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    img = plt.imread('ab.jpg')
    print(img.shape)
    grayscale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)
    _,binaryImg = cv2.threshold(grayscale,127,255,cv2.THRESH_BINARY)
    
    '''Structuring Elements'''
    
    structuringElement = np.ones((5,5), np.uint8)
    print('Structuring Element:','{}'.format(structuringElement))
    
    erosionImg = cv2.erode(binaryImg,structuringElement,iterations=1)
    process_img1 = erosion(binaryImg,structuringElement)
    dilationImg = cv2.dilate(binaryImg,structuringElement,iterations=1)
    process_img2 = dilation(binaryImg,structuringElement)
    openingImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN,structuringElement)
    process_img3 = opening(binaryImg,structuringElement)
    closingImg = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE,structuringElement)
    process_img4 = closing(binaryImg,structuringElement)
    
    
    
    img_set = [binaryImg,erosionImg,process_img1,dilationImg,process_img2,openingImg,process_img3,closingImg,process_img4]
    title_set = ['Binary Image','Built-in Erosion Image','Custom Erosion Image','Dilation Image','Dilation Image-2','Opening Image','Opening Image-2','Closing Image','Closing Image-2']
    

    
    img_plt(img_set,title_set)     

def erosion(binaryImg,structuringElement):
    r,c = binaryImg.shape
    m,n =  structuringElement.shape
    padding = (m-1)//2
    process_img = np.zeros((r,c),dtype=np.uint8)
    binaryImg = np.pad(binaryImg,padding,constant_values=0)
    for i in range(r):
        for j in range(c):
            res = np.sum(binaryImg[i:i+m,j:j+n]*structuringElement)
            if res == 255 * (m*n):
                process_img[i,j] =  255
    return process_img

def dilation(binaryImg,structuringElement):
    r,c = binaryImg.shape
    m,n =  structuringElement.shape
    padding = (m-1)//2
    process_img = np.zeros((r,c),dtype=np.uint8)
    binaryImg = np.pad(binaryImg,padding,constant_values=0)
    
    for i in range(r):
        for j in range(c):
            res = np.sum(binaryImg[i:i+m,j:j+n]*structuringElement)
            if res > 0:
                process_img[i,j] =  255
    return process_img
def opening(binaryImg,structuringElement):
    process_img1 = erosion(binaryImg,structuringElement)
    process_img2 = dilation(process_img1,structuringElement)
    return process_img2

def closing(binaryImg,structuringElement):
    process_img1= dilation(binaryImg,structuringElement)
    process_img2 = erosion(process_img1,structuringElement)
    return process_img2


def img_plt(img_set,title_set):
    ln = len(img_set)
    plt.figure(figsize=(20,20))
    for i in range(ln):
        ch = len(img_set[i].shape) 
        plt.subplot(3,3,i+1)
        if ch == 3 :
            plt.imshow(img_set[i])
        else :
            plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
    plt.show();
        
    
if __name__  == '__main__':
    main()
    