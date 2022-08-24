import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    img_path = "dog.jpg"
    rgbImg = plt.imread(img_path)
    print(rgbImg.shape)
    grayscale = cv2.cvtColor(rgbImg,cv2.COLOR_BGR2GRAY)
    r,c = grayscale.shape
    
    #Average Filtered
    avgKernel = np.ones((3, 3)) / 9
    print('avgKernel: {}'.format(avgKernel))
    filteredImg = cv2.filter2D(grayscale, -1, avgKernel)
    
    #Noisy Image
    noisyImg = np.copy(grayscale)
    rang = np.random.randint(5000,8000)
    print(rang)

    for i in range(rang):
        bit = np.random.randint(0,2)
        x,y = np.random.randint(0,(r,c))
        noisyImg[x,y] = bit*255
        
    """Filtered Noisy Image with Avergaing Kernel"""  
    avgFilteredNoisyImg = cv2.filter2D(noisyImg, -1, avgKernel)
    
    """Filtered Noisy Image with Gaussian Kernel"""     
    gaussianKernel = np.array([[1/16,1/8,1/16],
                              [1/8,1/4,1/8],
                              [1/16,1/8,1/16]])
    print('gaussianKernel: {}'.format(gaussianKernel))
    gaussianFilteredNoisyImg = cv2.filter2D(noisyImg, -1, gaussianKernel)
    
    """Filtered Noisy Image with Median Blur""" 
    medianFilteredNoisyImg  = cv2.medianBlur(noisyImg,3)
  
    img_set = [grayscale, filteredImg, noisyImg,avgFilteredNoisyImg,gaussianFilteredNoisyImg,medianFilteredNoisyImg]
    title_set = ['Grayscale', 'Filtered Image\n(Averaging)', 'Noisy Image\n(Salt & Pepper Noise)','Filtered Noisy Image\n(Averaging)','Filtered Noisy Image\n(Gaussian Kernel)','Filtered Noisy Image\n(Median Filter)']
    plot_img(img_set, title_set)

def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig("figures.jpg")
    plt.show()		
	
if __name__ == "__main__":
    main()