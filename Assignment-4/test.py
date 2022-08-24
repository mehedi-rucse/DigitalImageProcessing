import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.util import img_as_float


def main():
    img_path = 'Dog2.jpg'
    rgbImg = plt.imread(img_path)
    print(rgbImg.shape)
    grayscale = cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
    
    """used for converting the pixel values into float 
    as there was type mismatch betweet the ketnel and image"""
    
    grayscale = img_as_float(grayscale)
    # grayscale = np.zeros((3, 3), dtype=np.float32) 
    # grayscale += np.arange(3)
    print(grayscale.shape)
    
    '''Generating Kernels'''
    #kernel1 = np.ones((3, 3), dtype=np.float32) *(1)/3
    kernel1 = np.array([[0, -1, 0],
                        [0, 4, 0],
                        [0, -1, 0]])
    print('Kernel1: {}'.format(kernel1))
    
    x, y = grayscale.shape
    kx,ky = kernel1.shape
    r = x + kx - 1
    c = y + ky - 1
    padded_img = np.zeros((r,c),dtype=np.float32)
    
    """Zero padding the origianl image"""
    
    for i in range(x):
        for j in range(y):
            padded_img[i+(kx-1)//2,j+(ky-1)//2] = grayscale[i,j]
            
    processed_img1 = np.zeros((x,y),dtype=np.float32)

    """Replacing the value with kernel* matrix value """
    
    for i in range(r):
        for j in range(c): 
            for k in range(kx):
                for l in range(kx): 
                    if i < x and i >= 0 and j < y and j >= 0: 
                        processed_img1[i,j] += kernel1[k,l] * padded_img[i+k,j+l] 
                        if(processed_img1[i,j]) >= 256:
                            processed_img1[i,j] = 255
        
        
    """Filtering with built in function"""  
              
    processed_img2 = cv2.filter2D(grayscale,-1,kernel1);

    plt.figure(figsize=(20,20))
    plt.subplot(1,3,1)
    plt.imshow(grayscale,cmap='gray')
    plt.title("Original Grayscale Image")
    plt.subplot(1,3,2)
    plt.imshow(processed_img1,cmap='gray')
    plt.title("Manually Filtered Image with zero padding")
    plt.subplot(1,3,3)
    plt.imshow(processed_img2,cmap='gray')
    plt.title("Automated Filtered Image")
    plt.savefig('filtering.jpg')
    plt.show()
    
if __name__ == '__main__':
    main()
