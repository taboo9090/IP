# main.py

import Generics.generics as generics
import Segmentation.segmentation as segmentation
import cv2

image_grey = cv2.imread("/home/taboo/dev/temp/Lena-grey.jpeg")
image_colored = cv2.imread("/home/taboo/dev/temp/Lena-color.jpeg")

def show(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    pass

if __name__ == "__main__":
    main()