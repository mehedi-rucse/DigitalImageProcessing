import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float

def convolution(kernel,grayscale):
    x, y = grayscale.shape
    print(x,y)
    kx,ky = kernel.shape
    print(kx,ky)
    r = x + kx - 1
    c = y + ky - 1
    padded_img = np.zeros((r,c),dtype=np.float32)
    
    """Zero padding the origianl image"""
    
    for i in range(x):
        for j in range(y):
            padded_img[i+(kx-1)//2,j+(ky-1)//2] = grayscale[i,j]
            
    processed_img = np.zeros((x,y),dtype=np.float32)
    for i in range(r):
        for j in range(c): 
            for k in range(kx):     
                for l in range(ky): 
                    if i < x and i >= 0 and j < y and j >= 0: 
                        processed_img[i,j] += kernel[k,l] * padded_img[i+k,j+l] 
                        if(processed_img[i,j]) >= 256:
                            processed_img[i,j] = 255
    return processed_img


def main():
    img_path = 'bird.png'
    rgbImg = plt.imread(img_path)
    print(rgbImg.shape)
    grayscale = cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
    
    """
    used for converting the pixel values into float 
    as there was type mismatch betweet the ketnel and image
    """
    
    grayscale = img_as_float(grayscale)
    print(grayscale.shape)
    
    '''Generating Kernels'''

    lapacianKernel = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    print('lapacianKernel: {}'.format(lapacianKernel))
    
    sobelKernel = kernel = np.array([
                                    [-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]
                                    ])
    print('SobelKernel: {}'.format(sobelKernel))
    
    
        
    lapacianImg1 = convolution(lapacianKernel,grayscale) 
    sobelImg1 = convolution(sobelKernel,grayscale)   
    
    """Filtering with built in function"""            
    lapacianImg2 = cv2.filter2D(grayscale,-1,lapacianKernel);
    sobelImg2 = cv2.filter2D(grayscale,-1,sobelKernel);


    img_set = [grayscale, lapacianImg1, lapacianImg2, grayscale, sobelImg1, sobelImg2]
    title_set = ['Original Image', 'Manually Filtered lapacian', 'Automated Filtered lapacian','Original Image', 'Manually Filtered sobel', 'Automated Filtered sobel']
    plot_img(img_set,title_set)
    
    
def plot_img(img_set,title_set):
    n= len(img_set)
    plt.figure(figsize=(20,20))
    for i in range(n):
        plt.subplot(2,3,i+1)
        plt.imshow(img_set[i],cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig('Convolution.jpg')
    plt.show()
    
if __name__ == '__main__':
    main()