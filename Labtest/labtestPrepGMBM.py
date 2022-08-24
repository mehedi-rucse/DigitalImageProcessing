'''
1.Grayscale, Binary Convert
2.Manual Histogram
3.Bit masking and Bit slicing
4.Move Intensity
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = "/home/mehedi/Desktop/Digital Image Processing/Pic/Lionel.jpeg"
    rgbImg = plt.imread(path)
    rgbImg1 = cv2.resize(rgbImg,(200,200))
    print(rgbImg.shape)
    
    grayscale = cvtGrayscale(rgbImg)
    r,c = grayscale.shape
    
    #binary = cvtBinary(grayscale)
     #_,binary = cv2.threshold(grayscale,127,255,cv2.THRESH_BINARY)
    
    #movedIntensityImg = move_intensity(grayscale,50)
    
   # bitSlicing(grayscale)
    
   # bitMasking(grayscale)
    
    img_set = [rgbImg,rgbImg1]
    title_set = ['rgbImg','grayscale']
    
    plt_img(img_set,title_set)
    #plt_hist(img_set,title_set,r,c)
    
def cvtGrayscale(img):
    r,c,k = img.shape
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    grayscale = np.zeros((r,c),dtype=np.uint8)
    
    for i in range(r):
        for j in range(c):
            temp= 0.144 * red[i,j] + 0.587 * green[i,j] + 0.299 * blue[i,j]
            temp = max(temp,0)
            temp = min(255,temp)
            grayscale[i,j] =  temp
    return grayscale

def cvtBinary(img):
    r,c = img.shape
    binary = np.zeros((r,c),dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            if img[i,j] > 127:
                binary[i,j] = 255
    return binary

def move_intensity(img,value):
    r,c = img.shape
    processed_img = np.zeros((r,c),dtype=np.uint8)
    for i in range(r):
        for j in range(c):
                temp = img[i,j] + value
                temp = min(255,temp)
                temp = max(0,temp)
                processed_img[i,j] = temp
    return processed_img

def bitMasking(grayscale):
    
    plt.figure(figsize=(20,20))
    r,c = grayscale.shape
    black_img = np.zeros((r,c),dtype=np.uint8)

    mask = cv2.circle(black_img,(r//2,c//2),300,(255,255,255),-1)
    imgAnd =  cv2.bitwise_and(grayscale.copy(),mask)
    imgOr =  cv2.bitwise_or(grayscale.copy(),mask)
    imgXor =  cv2.bitwise_xor(grayscale.copy(),mask)
    imgNot =  cv2.bitwise_not(grayscale.copy())
    
    img_set = [grayscale,mask,imgAnd,imgOr,imgXor,imgNot]
    title_set = ['grayscale','mask','imgAnd','imgOr','imgXor','imgNot']
    ln = len(img_set)
    for i in range(ln):
        plt.subplot(2,3,i+1) 
        plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
    plt.show()
    
def bitSlicing(grayscale):
    r,c = grayscale.shape
    plt.figure(figsize=(20,20))
    for i in range (8):
        bit = np.zeros((r,c),np.uint8)
        for j in range(r):
            for k in range(c):
                bit[j,k] = grayscale[j,k] & pow(2,i)
        plt.subplot(2,4,i+1) 
        plt.title('bit :'+str(i))
        plt.imshow(bit,cmap='gray')
    plt.show()
            
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
    l=0
    for i in range(ln):
        plt.subplot(1,2,l+1) 
        ch = len(img_set[i])
        if ch==3:
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
        l = l+1
    plt.show()                    
if __name__ == "__main__":
    main()
    