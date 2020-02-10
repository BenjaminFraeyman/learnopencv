import cv2
import math

def fillHoles(mask):
    '''
        This hole filling algorithm is decribed in this post
        https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    '''
    maskFloodfill = mask.copy()
    h, w = maskFloodfill.shape[:2]
    maskTemp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(maskFloodfill)
    return mask2 | mask

def ellipse_X(width: int, height: int, y: int) -> int:
    '''
        Retrieve a certain x value on the ellipse for a given y.
    '''
    a = width / 2
    b = height / 2

    y_origin = b - y

    return int(math.sqrt(
        1 * (math.pow(a, 2) * (1 - math.pow(y_origin, 2) / math.pow(b, 2)))
    ))

def ellipse_Y(width: int, height: int, x: int) -> int:
    '''
        Retrieve a certain Y value on the ellipse for a given x.
    '''
    a = width / 2
    b = height / 2

    x_origin = a - x

    return int(math.sqrt(
        1 * (math.pow(b, 2) * (1 - math.pow(x_origin, 2) / math.pow(a, 2)))
    ))