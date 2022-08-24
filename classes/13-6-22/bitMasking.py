import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path1 = 'Dog2.jpg'
    Img1 = plt.imread(img_path1)  
    grayscale = cv2.cvtColor(Img1,cv2.COLOR_RGB2GRAY)
    #grayscale = np.ones((5,5),dtype=np.uint8)
    print(grayscale.shape) 
    r,c = grayscale.shape  
    print(grayscale)
    bit = 1
    pos=1
    plt.figure(figsize=(20,20))
    bits = np.zeros((r,c),dtype=np.uint8)
    for k in range(8):
        for i in range(r):
            for j in range(c):
                bits[i,j] = grayscale[i,j] & bit
                if bit == 128:
                    break
        plt.subplot(2,4,pos)
        plt.title("Bit "+str(pos)+" Slice")
        print(bits)
        plt.imshow(bits,cmap='gray')
        pos=pos+1
        bit = bit*2
    plt.show()        
    
if __name__ == '__main__':
    main()

            