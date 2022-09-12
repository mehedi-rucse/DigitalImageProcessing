#Convert files from jpeg to png and vice versa
import cv2
def main():
    img_path = "img.jpg"
    #img_path = "bird.png"
    name,format = img_path.split(".")
    if format == 'jpeg' or format == 'jpg':
        img = cv2.imread(img_path)
        cv2.imwrite(name+".png", img)
    else:
        img = cv2.imread(img_path)
        cv2.imwrite(name+".jpeg", img)
if __name__ == "__main__":
    main()
