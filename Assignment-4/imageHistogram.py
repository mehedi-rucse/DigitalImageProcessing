import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = "parrot.jpg"
    rgbImg = plt.imread(img_path)
    print(rgbImg.shape)
    
    i,j,k = rgbImg.shape
    image1 = np.zeros((256,), dtype=int)
    
    for x in range(i):
        for y in range(j): 
            tmp = rgbImg[x,y]
            image1[tmp] += 1
            
    y =  np.arange(256)
    
    plt.figure(figsize=(20,20))
    plt.subplot(1,3,1)
    plt.title('Image')
    plt.imshow(rgbImg)
    plt.subplot(1,3,2)
    plt.hist(rgbImg.ravel(),256,[0,256]) 
    plt.title('Histogram using built-in function')
    
    plt.subplot(1,3,3)
    plt.plot(y,image1)
    plt.ylim(0,)
    plt.fill_between(y,image1)
    plt.title("Histogram implemented manually")
    plt.savefig('histograms.jpg')
    plt.show()

if __name__ == '__main__':
    main()