import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path1 = 'dora.jpg'
    Img1 = plt.imread(img_path1)  
    grayscale = cv2.cvtColor(Img1,cv2.COLOR_RGB2GRAY)
    print(grayscale.shape) 
    r,c = grayscale.shape  
    bit = 1
    pos=2
    plt.figure(figsize=(20,20))
    plt.subplot(3,3,1)
    plt.imshow(grayscale,cmap="gray")
    bits = np.zeros((r,c),dtype=np.uint8)
    for k in range(8):
        for i in range(r):
            for j in range(c):
                bits[i,j] = grayscale[i,j] & bit
        plt.subplot(3,3,pos)
        plt.title("Bit "+str(pos-1)+" Slice")
        plt.imshow(bits,cmap='gray')
        pos=pos+1
        bit = bit*2
    plt.savefig('slice.jpg')
    plt.show()        
    
if __name__ == '__main__':
    main()

            