'''Module related to slicing image'''
from .utils import crop

__all__ = [
    'get_sliced_images',
]


def _get_sliced_image(image, i, j, config):
    '''Returns a single sliced image'''
    x = config['x']
    y = config['y']
    x_gap = config['x_gap']
    y_gap = config['y_gap']
    width = config['width']
    height = config['height']
    rect = x+i*x_gap, y+j*y_gap, width, height
    rect = tuple(int(a) for a in rect)
    roi = crop(image, rect)
    return rect, roi


def get_sliced_images(image, config):
    '''Returns sliced images'''
    sliced_images = {}
    x_count = config['x_count']
    y_count = config['y_count']
    for y in range(y_count):
        for x in range(x_count):
            rect, sliced_image = _get_sliced_image(image, x, y, config)
            sliced_images[rect] = sliced_image
    return sliced_images
