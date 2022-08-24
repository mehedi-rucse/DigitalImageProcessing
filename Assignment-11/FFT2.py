import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    rgbImg = plt.imread('dahlia.jpg')
    print(rgbImg.shape)
    gray =  cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
    ftImg = np.fft.fft2(gray)
    centeredfti_img = np.fft.fftshift(ftImg)
    magnitude_spectrum = 100 * np.log(np.abs(ftImg))
    centered_magnitude_spectrum = 100 * np.log(np.abs(centeredfti_img))
    
    r , c = gray.shape
    m , n = r//2 , c//2
    black_img = np.zeros((r,c),dtype=np.uint8)
    white_img = np.ones((r,c),dtype=np.uint8)
    
    filter1 = cv2.circle(black_img.copy(),(m,n),25,(255,255,255),-1)
    filter2 = cv2.circle(white_img.copy(),(m,n),25,(0,0,0),-1)
    filter3 = cv2.line(black_img.copy(),(m,0),(m,c),(255,255,255),9)
    filter4 = cv2.line(white_img.copy(),(0,n),(r,n),(0,0,0),9)

    
    ftimg_gf = centeredfti_img * filter4
    filtered_img = np.abs(np.fft.ifft2(ftimg_gf))

    
    img_set = [rgbImg, gray, magnitude_spectrum, centered_magnitude_spectrum, filter4, filtered_img]
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
