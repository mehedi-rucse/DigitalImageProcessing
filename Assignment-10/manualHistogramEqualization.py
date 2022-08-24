from pickletools import uint8
import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
    path = "/home/mehedi/Desktop/Digital Image Processing/Pic/dahlia.jpg"
    rgbImg = plt.imread(path)
    print(rgbImg.shape)
    grayscale = cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
    r,c = grayscale.shape
    equalizedImg = cv2.equalizeHist(grayscale)
    manEqualizedImg =  hist_equalize(grayscale)
    
    img_set = [rgbImg,grayscale,equalizedImg,manEqualizedImg]
    title_set = ['rgbImg','grayscale','equalizedImg','manEqualizedImg']
    
    plt_img(img_set,title_set)
    plt_hist(img_set,title_set,r,c)
    
def hist_equalize(img):
    L = 256
    histogram =  cv2.calcHist([img],[0],None,[256],[0,256])
    CDF = histogram.cumsum()
    CDF_min = CDF.min()
    r,c = img.shape
    size = r * c
    processed_img = np.zeros((r,c),dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            processed_img[i,j] = ((CDF[img[i,j]] - CDF_min) / (size - CDF_min)) * L-1
    return processed_img
            
        
    
def plt_hist(img_set,title_set,r,c):
    ln = len(img_set)
    l = 4
    for i in range(ln):
        plt.subplot(2,4,l+1) 
        img = img_set[i]
        histogram = np.zeros((256,),dtype=int)
        y = np.arange(256)
        for j in range(r):
            for k in range(c):
                temp = img[j,k]
                histogram[temp] +=1
        plt.plot(y,histogram)
        plt.ylim(0,)
        plt.title(title_set[i])
        l = l+1
    plt.show()

def plt_img(img_set,title_set):
    ln = len(img_set)
    plt.figure(figsize=(20,20))
    l = 0
    for i in range(ln):
        plt.subplot(2,4,l+1) 
        ch = len(img_set[i])
        if ch==3:
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
        l = l+1


                            
if __name__ == "__main__":
    main()
    