import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = "pic.jpg"
    grayscale = plt.imread(path)
    print(grayscale.shape)
    r,c = grayscale.shape
    #_,binary = cv2.threshold(grayscale,127,255,cv2.THRESH_BINARY)
    image1 = cv2.equalizeHist(grayscale)
    image2 = histEqualization(grayscale)
    
    
    img_set = [grayscale,image1,image2]
    title_set = ['grayscale','Equalized','Man Equalized']
    
    plt_img(img_set,title_set)
    plt_hist(img_set,title_set,r,c)
    
def histEqualization(grayscale):
    L=256
    histogram = cv2.calcHist([grayscale],[0],None,[256],[0,256])
    CDF = histogram.cumsum()
    CDF_min = CDF.min()
    r,c = grayscale.shape
    process_img = np.zeros((r,c),np.uint8)
    size = r*c
    for i in range(r):
        for j in range(c):
            process_img[i,j] = ((CDF[grayscale[i,j]] - CDF_min)/(size-CDF_min)) * L-1
    return process_img
    
def plt_hist(img_set,title_set,r,c):
    ln = len(img_set)
    l = 3
    for i in range(ln):
        plt.subplot(2,3,l+1) 
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
    l=0
    for i in range(ln):
        plt.subplot(2,3,l+1) 
        ch = len(img_set[i])
        if ch==3:
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
        l = l+1
                            
if __name__ == "__main__":
    main()