import matplotlib.pyplot as plt
import cv2
img_path = '/home/mehedi/Desktop/Digital Image Processing/Pic/parrot.jpg'

def main():
    
    rgbImg = plt.imread(img_path)

    redImg = rgbImg[:,:,0]
    greenImg = rgbImg[:,:,1]
    blueImg = rgbImg[:,:,2]
    
    plt.figure(figsize=(20,20))
    _, binaryImg =  cv2.threshold(rgbImg, 127, 255, cv2.THRESH_BINARY)
    grayImg =  cv2.cvtColor(rgbImg,cv2.COLOR_RGB2GRAY)

    
    plt.subplot(3,4,1)
    plt.imshow(rgbImg)
    plt.title('Original Image')
    plt.subplot(3,4,2)
    plt.hist(rgbImg.ravel(),256,[0,256]) 
    plt.title('Original Image Histogram')
    
    
    plt.subplot(3,4,3)
    plt.imshow(redImg,cmap='gray')
    plt.title('Red')
    plt.subplot(3,4,4)
    plt.hist(redImg.ravel(),256,[0,256])
    plt.title('Red channel Histogram')
   
    
    plt.subplot(3,4,5)
    plt.imshow(greenImg,cmap='gray')
    plt.title('Green')
    plt.subplot(3,4,6)
    plt.hist(greenImg.ravel(),256,[0,256]) 
    plt.title('Green channel Histogram')
  
    
    plt.subplot(3,4,7)
    plt.imshow(blueImg,cmap='gray')
    plt.title('Blue')
    plt.subplot(3,4,8)
    plt.hist(blueImg.ravel(),256,[0,256]) 
    plt.title('Blue channel Histogram')
    
    
    plt.subplot(3,4,9)
    plt.imshow(binaryImg,cmap='gray')
    plt.title('Binary')
    plt.subplot(3,4,10)
    plt.hist(binaryImg.ravel(),256,[0,256]) 
    plt.title('Binary channel Histogram')
    
    
    plt.subplot(3,4,11)
    plt.imshow(grayImg,cmap='gray')
    plt.title('Gray')
    plt.subplot(3,4,12)
    plt.hist(grayImg.ravel(),256,[0,256]) 
    plt.title('Grayscale Histogram')
    
    plt.show()
    plt.close()
    


if __name__ == '__main__':
    main()
    