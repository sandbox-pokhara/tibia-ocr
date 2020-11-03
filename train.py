'''Main module'''
import json

import cv2

from base.screen import crop
from base.screen import get_hash

letters = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')


def main(debug=False):
    '''Main function'''
    image = cv2.imread('train.png')
    threshed = cv2.inRange(image, (244, 244, 244), (244, 244, 244))
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    model = []
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        letter_image = crop(threshed, bounding_rect)
        letter_hash = get_hash(letter_image)
        letter = letters.pop(0)
        if debug:
            cv2.imshow(letter, cv2.resize(letter_image, None, fx=30, fy=30, interpolation=0))
            cv2.waitKey()
            cv2.destroyWindow(letter)
        model.append({
            'letter': letter,
            'hash': letter_hash,
            'width': bounding_rect[2],
            'height': bounding_rect[3],
        })
    with open('letters.json', 'w') as file_pointer:
        json.dump(model, file_pointer, indent=2)


if __name__ == '__main__':
    main()
