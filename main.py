'''Main module'''
import cv2

from tibia.hash_ocr import convert_line
from tibia.hash_ocr import convert_paragraph


def main():
    '''Main function'''
    image = cv2.imread('data/test.png')
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    print(convert_line(threshed))

    image = cv2.imread('data/test1.png')
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    print(convert_line(threshed))

    image = cv2.imread('data/test2.png')
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    print(convert_line(threshed))

    image = cv2.imread('data/test3.png')
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    print(convert_paragraph(threshed))

    image = cv2.imread('data/test4.png')
    threshed = cv2.inRange(image, (244, 244, 244), (244, 244, 244))
    print(convert_line(threshed))

    image = cv2.imread('data/test5.png')
    threshed = cv2.inRange(image, (191, 191, 191), (191, 191, 191))
    print(convert_line(threshed))


if __name__ == '__main__':
    main()
