import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
    img_path = 'dora.jpg'
    Img1 = plt.imread(img_path)  
    Image1 = cv2.cvtColor(Img1,cv2.COLOR_RGB2GRAY)
    print(Image1.shape)

    Image2 = np.zeros(Image1.shape, dtype=np.uint8)
    Image2 = cv2.circle(Image2, (1000, 1300), 500, (255,255,255), -1) 
    
    print(Image2.shape)
    
    resultAnd = cv2.bitwise_and(Image2, Image1)
    resultOr = cv2.bitwise_or(Image2, Image1)
    resultXor = cv2.bitwise_xor(Image2, Image1)
    resultNot = cv2.bitwise_not(Image1)
    
    img_set = [Image1, Image2, resultAnd, resultOr,resultXor, resultNot]
    title_set = ['Image-1', 'Image-2', 'AND', 'OR','XOR', 'NOT']
    plot_img(img_set,title_set)
    
    
def plot_img(img_set,title_set):
    n= len(img_set)
    plt.figure(figsize=(20,20))
    for i in range(n):
        img = img_set[i]
        ch = len(img.shape)
        plt.subplot(2,3,i+1)
        plt.imshow(img_set[i],cmap = 'gray')
        plt.title(title_set[i])
    plt.savefig('masks.jpg')
    plt.show()
    
if __name__ == '__main__':
    main()
