'''
    Copyright 2017 by Satya Mallick ( Big Vision LLC )
    http://www.learnopencv.com
'''

import cv2
import numpy as np
import math
from pathlib import Path

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

if __name__ == '__main__' :
    cwd = Path.cwd()

    # Read image
    img = cv2.imread(str(Path("RedEyeRemover/Pictures/Bloodshot/edited4.jpg")), cv2.IMREAD_COLOR)
    
    # Output image
    imgOut = img.copy()
    
    # Load HAAR cascade
    # eyesCascade = cv2.CascadeClassifier(str(Path("RedEyeRemover/HaarCascades/default.xml")))
    eyesCascade = cv2.CascadeClassifier(str(Path("RedEyeRemover/HaarCascades/test1.xml")))
    
    minwidth = int(img.shape[1] / 7)
    minheight = int(img.shape[0] / 9)

    # Detect eyes
    # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
    # https://stackoverflow.com/questions/51132674/meaning-of-parameters-of-detectmultiscalea-b-c/51356792
    # https://stackoverflow.com/questions/22249579/opencv-detectmultiscale-minneighbors-parameter
    eyes = eyesCascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=8, minSize=(minwidth, minheight))
    
    # For every detected eye
    for (x, y, w, h) in eyes:
        # Make eye selection a bit smaller
        # y = y + 45
        # h = h - 70

        # Extract eye from the image
        eye = img[y:y+h, x:x+w]

        # cv2.imshow('Eye', eye)
        # cv2.waitKey(0)

        # Split eye image into 3 channels
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]

        r = r - 20

        # filter to detect redness
        f = eye[:, :, 0].astype(np.float)

        # https://ieeexplore.ieee.org/document/1038147
        for row, (blue_row, green_row, red_row) in enumerate(zip(b, g, r)):
            for column, (blue, green, red) in enumerate(zip(blue_row, green_row, red_row)):
                f[row, column] = float(math.pow(red, 2) / (math.pow(blue, 2) + math.pow(green, 2)))
                # print(f[row, column])

        # Add the green and blue channels.
        bg = cv2.add(b, g)

        # Simple red eye detector.
        #mask_higher = (r > 80) &  (r > bg)
        mask_higher = (f > 0.80) & (r > b) & (r > g)
        
        # Convert the mask to uint8 format.
        mask_higher = mask_higher.astype(np.uint8)*255

        # Clean mask -- 1) Fill holes 2) Dilate (expand) mask.
        # mask_higher = fillHoles(mask_higher)
        # mask_higher = cv2.dilate(mask_higher, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

        # Calculate the mean channel by averaging the green and blue channels
        mean = bg / 2
        mean = mean[:, :, np.newaxis]
        mask_higher = mask_higher.astype(np.bool)[:, :, np.newaxis]

        # Copy the eye from the original image.
        eyeOut = eye.copy()

        # Copy the mean image to the output image.
        # np.copyto(eyeOut, mean, where=mask)
        eyeOut_higher = np.where(mask_higher, mean, eyeOut)

        # Copy the fixed eye to the output image.
        imgOut[y:y+h, x:x+w, :] = eyeOut_higher

    # Display Result
    cv2.imshow('Red Eyes', img)
    cv2.imshow('Red Eyes Removed', imgOut)
    cv2.waitKey(0)
