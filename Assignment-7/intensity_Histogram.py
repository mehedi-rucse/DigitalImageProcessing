import cv2
import numpy as np
import matplotlib.pyplot as plt

def img_hist(img_set,hist_title_set):
    ch = len(img_set)
    j=2
    for i in range(ch):
        plt.subplot(2,4,j)
        plt.hist(img_set[i].ravel(),250,[0,250])
        plt.title(hist_title_set[i])
        j=j+2
    plt.savefig("intensity.jpg")    
    plt.show()
        
def plt_img(img_set,img_title):
    ch = len(img_set)
    plt.figure(figsize=(20,20))
    j=1
    for i in range(ch):
        plt.subplot(2,4,j)
        plt.imshow(img_set[i],cmap='gray')
        plt.title(img_title[i])
        j = j+2
        
def move_intensity(grayscale,d):
    x,y = grayscale.shape
    processed_img = np.zeros((x,y),np.uint8)

    for i in range(x):
        for j in range(y):  
            tmp =  grayscale[i,j] + d
            if (tmp > 255 ):
                tmp = 255
            if (tmp <0 ):
                tmp = 0
            processed_img[i,j] = tmp
    return processed_img

def band_intensity(grayscale,r1,r2):
    x,y = grayscale.shape
    processed_img = np.zeros((x,y),np.uint8)

    for i in range(x):
        for j in range(y):  
            tmp =  grayscale[i,j] 
            if (tmp < r1 ):
                tmp = r1
            if (tmp > r2 ):
                tmp = r2
            processed_img[i,j] = tmp
    return processed_img          
    
def main():
    rgbImg = plt.imread('field.jpg')
    print(rgbImg.shape)
    grayscale = cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)
    r,c = grayscale.shape
    
    movedLeft = np.zeros((r,c),dtype=np.uint8)
    movedRight = np.zeros((r,c),dtype=np.uint8)
    narrowBand = np.copy(grayscale)

    movedLeft = move_intensity(grayscale,90)
    movedRight = move_intensity(grayscale,-90)
    narrowBand = band_intensity(grayscale,100,150)
    
    img_set = [grayscale,movedLeft,movedRight,narrowBand]
    img_title = ['Grayscale','Moved Left','Moved Right','Narrow Band']
    hist_title_set = ['Grayscale Histogram', 'Left Histogram','Right Histogram','Band Histogram']

    plt_img(img_set,img_title)
    img_hist(img_set,hist_title_set)
   
if __name__ == '__main__':
    main()