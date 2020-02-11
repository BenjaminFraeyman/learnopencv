'''
    Copyright 2017 by Satya Mallick ( Big Vision LLC )
    http://www.learnopencv.com
'''

import cv2
import numpy as np
import math
from pathlib import Path
import redeyelib as rel
import defaults

if __name__ == '__main__' :
    # Read image
    # img = cv2.imread(str(Path("RedEyeRemover/Pictures/Bloodshot/irritated.png")), cv2.IMREAD_COLOR)
    img = cv2.imread(str(Path("RedEyeRemover/Pictures/Bloodshot/edited6.jpg")), cv2.IMREAD_COLOR)
    
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
        y = y + defaults.Y_TOP_REDUCTION
        h = h - defaults.Y_BOTTOM_REDUCTION
        x = x + defaults.X_LEFT_OFFSET
        w = w - defaults.X_RIGHT_OFFSET

        # Extract eye from the image
        eye = img[y:y+h, x:x+w]

        # Display eye
        # cv2.imshow('Eye', eye)
        # cv2.waitKey(0)

        # Split eye image into 3 channels
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]

        # Reduce redness overall
        r = r - defaults.RED_OFFSET

        # Calculate ellipse cutoffs
        cutoffs = []
        for x_value in range(0, int(w/2)):
            cutoffs.append([x_value, rel.ellipse_Y(w, h, x_value)])

        # Pixels that do not count
        non_discarded_pixels = 0

        for (cutoff_x, cutoff_y) in cutoffs:
            cutoff_y_upper = int(h/2) - cutoff_y
            cutoff_y_lower = int(h/2) + cutoff_y

            non_discarded_pixels = non_discarded_pixels + (4 * cutoff_y)

            for y_row in range(0, cutoff_y_upper):
                r[y_row][:cutoff_x] = 0
                r[y_row][w - cutoff_x:] = 0
                
            for y_row in range(cutoff_y_lower, h):
                r[y_row][:cutoff_x] = 0
                r[y_row][w - cutoff_x:] = 0

        print(f'non-discarded pixels: {non_discarded_pixels}')
        print(f'total pixels: {h * w}')
        print(f'percentage of discarded pixels: {1 - (non_discarded_pixels / (h * w))}')

        # Filter to detect redness
        f = eye[:, :, 0].astype(np.float)

        # https://ieeexplore.ieee.org/document/1038147
        for row, (blue_row, green_row, red_row) in enumerate(zip(b, g, r)):
            for column, (blue, green, red) in enumerate(zip(blue_row, green_row, red_row)):
                f[row, column] = float(math.pow(red, 2) / (math.pow(blue, 2) + math.pow(green, 2) + 1))

        # Add the green and blue channels together.
        bg = cv2.add(b, g)

        # Simple red eye detector.
        # mask_higher = (r > 80) &  (r > bg)
        mask_higher = (f > defaults.REDNESS_FILTER) & (r > b) & (r > g)
        
        # Convert the mask to uint8 format.
        mask_higher = mask_higher.astype(np.uint8)*255

        # Clean mask -- 1) Fill holes 2) Dilate (expand) mask.
        # mask_higher = rel.fillHoles(mask_higher)
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
