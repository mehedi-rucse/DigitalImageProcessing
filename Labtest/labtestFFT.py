import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = '/home/mehedi/Desktop/Digital Image Processing/Pic/dahlia.jpg'
    rgbImg = plt.imread(path)
    print(rgbImg.shape)
    grayscale = cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)
    r,c = grayscale.shape
    ftImg = np.fft.fft2(grayscale)
    centeredfti_img = np.fft.fftshift(ftImg)
    magnitude_spectrum = 100 * np.log(np.abs(ftImg))
    centered_magnitude_spectrum = 100 * np.log(np.abs(centeredfti_img))
    
    r , c = grayscale.shape
    m , n = r//2 , c//2
    black_img = np.zeros((r,c),dtype=np.uint8)
    white_img = np.ones((r,c),dtype=np.uint8)
    
    filter = cv2.circle(black_img.copy(),(m,n),25,(255,255,255),-1)
    ftiImg_gf =  centeredfti_img * filter
    filtered_img = np.abs(np.fft.ifft2(ftiImg_gf)) 
    
    img_set = [rgbImg, grayscale, magnitude_spectrum, centered_magnitude_spectrum, filter, filtered_img]
    title_set = ['RGB', 'Gray', 'FFT2', 'Centered FFT2', 'Filter', 'Filtered Img']
    
    plot_img(img_set,title_set)
    
def plot_img(img_set, title_set):		
    plt.figure(figsize = (20, 20))
    n = len(img_set)
    for i in range(n):
        plt.subplot(2, 3, i + 1)
        plt.title(title_set[i])
        img = img_set[i]
        ch = len(img.shape)
        if (ch == 2):
            plt.imshow(img, cmap = 'gray')
        else:
            plt.imshow(img)			
    plt.show()


if __name__ == '__main__':
    main()