import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = "parrot.jpg"
    rgbImg = plt.imread(img_path)
    print(rgbImg.shape)
    
    grayscale = cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)
    print(grayscale.shape)
    
    i, j = grayscale.shape
    processed_img1 = np.zeros((i,j), dtype = np.uint8)
    processed_img2 = np.zeros((i,j), dtype = np.uint8)
    processed_img3 = np.zeros((i,j), dtype = np.uint8)
    processed_img4 = np.zeros((i,j), dtype = np.uint8)
    
    '''Predefined Constant values'''
    
    epsilon = 0.0000001
    T1 = 20
    T2 = 80
    c = 2
    p = 5
    
    for x in range(i):
        for y in range(j):  
            processed_img3[x, y] = c * np.log(1+grayscale[x,y])
            processed_img4[x, y] = c * pow((grayscale[x,y] + epsilon),p)
            
            if (grayscale[x, y] >= T1 and grayscale[x, y] <= T2):
                processed_img1[x, y] = 100
                processed_img2[x, y] = 100
            else :
                processed_img1[x, y] = 10
                processed_img2[x, y] = grayscale[x,y]
            
    img_set = [rgbImg, grayscale, processed_img1, processed_img2,processed_img3, processed_img4]
    title_set = ['RGB', 'Grayscale', 'Condition-1', 'Condition-2','Condition-3', 'Condition-4']
    plot_img(img_set,title_set)
    
    
def plot_img(img_set,title_set):
    n= len(img_set)
    plt.figure(figsize=(20,20))
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(2,3,i+1)
        if (ch == 3):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig('figures.jpg') 
    plt.show()
    
        

if __name__  == '__main__' :
    main()