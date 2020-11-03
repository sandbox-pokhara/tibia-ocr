'''Main module'''
import json

import cv2

from base.decorators import load_once
from base.screen import crop
from base.screen import get_hash


@load_once
def _get_model():
    with open('letters.json') as file_pointer:
        letters = json.load(file_pointer)
        return {l['hash']: l['letter'] for l in letters}


def convert_letter(image):
    '''Ocr a letter'''
    model = _get_model()
    _, width = image.shape[:2]
    if width == 0:
        return ''
    for tentative_width in reversed(range(2, width+1)):
        letter_image = image[:, :tentative_width]
        contours, _ = cv2.findContours(letter_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours == []:
            continue
        letter_image = crop(letter_image, cv2.boundingRect(contours[0]))
        letter_hash = get_hash(letter_image)
        letter = model.get(letter_hash, None)
        if letter is not None:
            remaining_image = image[:, tentative_width:]
            return letter + convert_letter(remaining_image)
    return ''


def convert_line(image, debug=False):
    '''Ocr a line'''
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    line = ''
    for contour in contours:
        bounding_rect = cv2.boundingRect(contour)
        letter_image = crop(image, bounding_rect)
        letter = convert_letter(letter_image)
        line += letter
        if debug:
            cv2.imshow(letter, cv2.resize(letter_image, None, fx=30, fy=30, interpolation=0))
            cv2.waitKey()
            cv2.destroyWindow(letter)
    return line


def main():
    '''Main function'''
    image = cv2.imread('test.png')
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    print(convert_line(threshed))

    image = cv2.imread('test1.png')
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    print(convert_line(threshed))

    image = cv2.imread('test2.png')
    threshed = cv2.inRange(image, (192, 192, 192), (192, 192, 192))
    print(convert_line(threshed))


if __name__ == '__main__':
    main()
