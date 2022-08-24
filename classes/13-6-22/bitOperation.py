import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path1 = 'pic1.jpg'
    Img1 = plt.imread(img_path1)  
    Image1 = cv2.cvtColor(Img1,cv2.COLOR_RGB2GRAY)
    print(Image1.shape)
    
    img_path2 = 'img2.jpg'
    Img2 = plt.imread(img_path2) 
    Image2 = cv2.cvtColor(Img2,cv2.COLOR_RGB2GRAY)
    print(Image2.shape)
    
    resultAnd = cv2.bitwise_and(Image1, Image2, mask = None)
    resultOr = cv2.bitwise_or(Image1, Image2, mask = None)
    resultXor = cv2.bitwise_xor(Image1, Image2, mask = None)
    resultNot = cv2.bitwise_not(Image1, Image2, mask = None)
    
    
    img_set = [Image1, Img2, resultAnd, resultOr,resultXor, resultNot]
    title_set = ['Image-1', 'Image-2', 'And', 'Or','Xor', 'Not']
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
            plt.imshow(img_set[i],cmap = 'gray')
        plt.title(title_set[i])
    plt.show()
    
if __name__ == '__main__':
    main()
