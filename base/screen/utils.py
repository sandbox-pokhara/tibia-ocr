"""Utility functions related to screenshots"""
import hashlib
import math

import cv2
import numpy

__all__ = [
    "crop",
    "crop_to_content",
    "resize_to_content",
    "get_coordinates_from_image",
    "get_city_distance",
    "get_chessboard_distance",
    "get_offset",
    "get_difference",
    "get_closest",
    "get_hash",
]


def crop(img, rect):
    """Crops an cv2 image using coordinate and size"""
    return img[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]


def crop_to_content(image, threshed_image):
    """Crops an image to content using a threshed image"""
    contours, _ = cv2.findContours(
        threshed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    content = [-1, -1, -1, -1]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if content[0] == -1:
            content = [x, y, x + w, y + h]
        else:
            if x < content[0]:
                content[0] = x
            if y < content[1]:
                content[1] = y
            if x + w > content[2]:
                content[2] = x + w
            if y + h > content[3]:
                content[3] = y + h
    return image[content[1] : content[3], content[0] : content[2]]


def resize_to_content(image, width, height):
    """Resizes an image to content, adds padding if smaller than the content"""
    h, w = image.shape[:2]
    if h > height:
        image = image[:height]
    if h < height:
        to_append = height - h
        image = cv2.copyMakeBorder(
            image, 0, to_append, 0, 0, cv2.BORDER_CONSTANT
        )
    if w > width:
        image = image[:, :width]
    if w < width:
        to_append = width - w
        image = cv2.copyMakeBorder(
            image, 0, 0, 0, to_append, cv2.BORDER_CONSTANT
        )
    return image


def get_coordinates_from_image(file_path, color=(0, 255, 0)):
    """Converts image into coordinate list"""
    img = cv2.imread(file_path)
    points = numpy.where(numpy.all(img == color, axis=-1))
    points = list(zip(*points[::-1]))
    return points


def get_city_distance(coordinates_1, coordinates_2, z_inf=False):
    """Returns the 2d distance between two coordinates"""
    if coordinates_1[-1] != coordinates_2[-1] and z_inf:
        return math.inf
    x_1, y_1 = coordinates_1[:2]
    x_2, y_2 = coordinates_2[:2]
    return abs(y_2 - y_1) + abs(x_2 - x_1)


def get_chessboard_distance(coordinates_1, coordinates_2):
    """Returns the 2d distance between two coordinates"""
    x_1, y_1 = coordinates_1[:2]
    x_2, y_2 = coordinates_2[:2]
    return max(abs(y_2 - y_1), abs(x_2 - x_1))


def get_offset(coordinates_1, coordinates_2):
    """Returns the offset of one coordinates to another"""
    return tuple(
        a - b for a, b in zip(tuple(coordinates_1), tuple(coordinates_2))
    )


def get_difference(pixel_1, pixel_2):
    """Returns the difference between the two pixels"""
    return sum(abs(a - b) for a, b in zip(pixel_1, pixel_2))


def get_closest(reference_pixel, pixel_mapping):
    """Returns the closest color related to the reference_pixel"""
    score = {v: get_difference(reference_pixel, v) for v in pixel_mapping}
    closest_pixel = min(score, key=score.get)
    return pixel_mapping[closest_pixel]


def get_hash(image):
    """Returns the hash of the image"""
    return hashlib.md5(image.tobytes()).hexdigest()
