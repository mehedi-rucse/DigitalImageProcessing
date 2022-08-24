from concurrent.futures import process
import cv2
import numpy as np
import matplotlib.pyplot as plt
def main():
    img = plt.imread('mri1.jpg')
    print(img.shape)
    grayscale = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)
    r,c = grayscale.shape

    noisyImg = saltPepper(grayscale)
    # kernel1 = np.array([[-1, -1, -1],
    #                     [-1, 8, -1],
    #                     [-1, -1, -1]])
    
    kernel1 = np.array([[1/16,1/8,1/16],
                              [1/8,1/4,1/8],
                              [1/16,1/8,1/16]])
    
    process_img1 = cv2.filter2D(grayscale,-1,kernel1)
    process_img2 = convolution2D(grayscale,kernel1)

    img_set = [grayscale,noisyImg,process_img1,process_img2]
    title_set = ['Grayscale','noisyImg','Built-in Convolution','Manual Convolution']
    
    plt_img(img_set,title_set)
    plt_hist(img_set,title_set)

def saltPepper(grayscale):
    r,c = grayscale.shape
    noisyImg = np.copy(grayscale)
    rang = np.random.randint(5000,8000)
    for i in range(rang):
        bit =  np.random.randint(0,2)
        x,y = np.random.randint(0,(r,c))
        noisyImg[x,y] = bit * 255
    return noisyImg
        

def convolution2D(grayscale,kernel):
    r,c = grayscale.shape
    x,y = kernel.shape
    rx,cy = r+x-1,c+y-1
    padded_img = np.zeros((rx,cy),dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            padded_img[i+x//2,j+y//2] = grayscale[i,j] 
            
    # m = x//2
    #padded_img = np.pad(grayscale,m,constant_values=0)
    print(padded_img.shape)
    process_img = np.zeros((r,c),dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            temp = np.sum(padded_img[i:i+x,j:j+y]*kernel)
            temp = np.rint(temp)
            temp = max(temp,0)
            temp = min(255,temp)
            process_img[i,j] =  temp
            
    return process_img
            
    
def plt_hist(img_set,title_set):
    ln = len(img_set)
    l =  5
    for i in range(ln):
        plt.subplot(2,4,l)
        img = img_set[i]
        plt.hist(img.ravel(),256,[0,256])
        plt.title(title_set[i])
        l = l + 1	
    plt.show()

def plt_img(img_set,title_set):
    ln = len(img_set)
    plt.figure(figsize=(20,20))
    l = 1
    for i in range(ln):
        plt.subplot(2,4,l)
        plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
        l = l + 1

	
	
if __name__ == '__main__':
	main()
