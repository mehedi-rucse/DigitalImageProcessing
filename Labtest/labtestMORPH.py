import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = '/home/mehedi/Desktop/Digital Image Processing/Assignment-7/field.jpg'
    img = plt.imread(path)
    print(img.shape)
    grayscale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)
    _,binaryImg = cv2.threshold(grayscale,127,255,cv2.THRESH_BINARY)
    
    '''Generating Structuring Elements'''
    structElem1 = np.ones((3,3),dtype=np.uint8)
    erosionImg = cv2.erode(binaryImg,structElem1,iterations=1)
    process_img1 = erosion(binaryImg,structElem1)
    dilationImg = cv2.dilate(binaryImg,structElem1,iterations=1)
    process_img2 = dilation(binaryImg,structElem1)
    openingImg = cv2.morphologyEx(binaryImg, cv2.MORPH_OPEN,structElem1)
    process_img3 = opening(binaryImg,structElem1)
    closingImg = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE,structElem1)
    process_img4 = closing(binaryImg,structElem1)
    
    img_set = [binaryImg,erosionImg,process_img1,dilationImg,process_img2,openingImg,process_img3,closingImg,process_img4]
    title_set = ['Binary Image','Built-in Erosion Image','Custom Erosion Image','Dilation Image','Dilation Image-2','Opening Image','Opening Image-2','Closing Image','Closing Image-2']
    
    img_plt(img_set,title_set)    
    

def erosion(binaryImg,structElem1):
    r,c = binaryImg.shape
    x,y = structElem1.shape
    m = x // 2
    paddedImg = np.pad(binaryImg,1,constant_values=0)
    processImg = np.zeros((r,c),dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            temp = np.sum(paddedImg[i:i+x,j:j+y] * structElem1)
            if temp == 255 * (x*y) :
                processImg[i,j] = 255
    return processImg

def dilation(binaryImg,structElem1):
    r,c = binaryImg.shape
    x,y = structElem1.shape
    m = x // 2
    paddedImg = np.pad(binaryImg,1,constant_values=0)
    processImg = np.zeros((r,c),dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            temp = np.sum(paddedImg[i:i+x,j:j+y] * structElem1)
            if temp >0 :
                processImg[i,j] = 255
    return processImg 

def opening(binaryImg,structElem1):
    process_img1 = erosion(binaryImg,structElem1)
    process_img2 = dilation(process_img1,structElem1)
    return process_img2      
def closing(binaryImg,structElem1):
    process_img1 = dilation(binaryImg,structElem1)
    process_img2 = erosion(process_img1,structElem1)
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
    