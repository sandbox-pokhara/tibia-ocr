'''Module for template matching'''
import cv2
import numpy

__all__ = ['template_match']


def template_match(image, template, threshold=0.03, debug=False):
    '''Template match using template label'''
    result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
    locations = numpy.where(result <= threshold)
    locations = list(zip(*locations[::-1]))
    if debug:
        canvas = image.copy()
        for location in locations:
            cv2.circle(canvas, location, 5, (0, 255, 0), thickness=2)
        cv2.imshow('', canvas)
        cv2.waitKey()
    return locations
