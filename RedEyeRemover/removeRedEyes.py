'''
    Copyright 2017 by Satya Mallick ( Big Vision LLC )
    http://www.learnopencv.com

    target recognition: https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
'''

import cv2
import numpy as np
import math
from pathlib import Path
import imutils

import redeyelib as rel
import defaults
import logging

if __name__ == '__main__':
    logging.initiate()
    
    logging.tell(f'---- Loading image and target. ----')

    # Read image
    FACE = cv2.imread(str(defaults.INPUT_IMAGE), cv2.IMREAD_COLOR)
    FACE_GRAY = cv2.cvtColor(FACE, cv2.COLOR_BGR2GRAY)
    FACE_CANNY = cv2.Canny(FACE_GRAY, 50, 200)

    # Read the colourtarget
    TARGET = cv2.imread(str(defaults.TARGET_IMAGE), cv2.IMREAD_COLOR)
    TARGET_GRAY = cv2.cvtColor(TARGET, cv2.COLOR_BGR2GRAY)
    TARGET_CANNY = cv2.Canny(TARGET_GRAY, 50, 200)
    (tH, tW) = TARGET.shape[:2]

    # Bookkeeping variable to keep track of the matched region
    found = None

    logging.tell(f'---- Starting targetdetection. ----')

    # loop over the scales of the image
    for scale_h in np.linspace(0.2, 3, 50)[::-1]:
        for scale_w in np.linspace(0.2, 3, 50)[::-1]:
            # Resize the image according to the scale
            resized = cv2.resize(
                FACE_CANNY, 
                (
                    int(FACE.shape[1] * scale_w), 
                    int(FACE.shape[0] * scale_h)
                ),
                interpolation = cv2.INTER_AREA
            )

            # If the resized image is smaller than the template, then break from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            
            # Keep track of the ratio of the resizing
            r_w = FACE.shape[1] / float(resized.shape[1])
            r_h = FACE.shape[0] / float(resized.shape[0])

            # cv2.imshow("resized", resized)
            # cv2.waitKey(0)

            # Apply template matching to find the template in the image
            result = cv2.matchTemplate(resized, TARGET_CANNY, cv2.TM_CCOEFF)

            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then update the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r_w, r_h)

    logging.tell(f'\tTarget found: {found}')

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r_w, r_h) = found

    (startX, startY) = (
        int(maxLoc[0] * r_w), 
        int(maxLoc[1] * r_h)
    )
    (endX, endY) = (
        int((maxLoc[0] + tW) * r_w), 
        int((maxLoc[1] + tH) * r_h)
    )

    logging.tell(f'\tStart x/y: {startX} / {startY}')
    logging.tell(f'\tEnd x/y: {endX} / {endY}')

    # Draw a bounding box around the detected result and display the image
    FACE_TARGET = FACE.copy()
    cv2.rectangle(
        FACE_TARGET, 
        (startX, startY), 
        (endX, endY), 
        (0, 0, 255), 
        2
    )
    if defaults.DISPLAY:
        cv2.imshow("Target detected", FACE_TARGET)
        cv2.waitKey(0)
    
    logging.tell(f'---- Start of eyedetection. ----')

    # Output image
    FACE_OUT = FACE.copy()
    
    # Load HAAR cascade
    eyesCascade = cv2.CascadeClassifier(str(defaults.CLASSIFIER_CASCADE))
    
    minwidth = int(FACE.shape[1] * defaults.EYE_W_RATIO)
    minheight = int(FACE.shape[0] * defaults.EYE_H_RATIO)

    # Detect eyes
    # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
    # https://stackoverflow.com/questions/51132674/meaning-of-parameters-of-detectmultiscalea-b-c/51356792
    # https://stackoverflow.com/questions/22249579/opencv-detectmultiscale-minneighbors-parameter
    eyes = eyesCascade.detectMultiScale(
        FACE, 
        scaleFactor=defaults.SCALEFACTOR, 
        minNeighbors=defaults.MIN_NEIGHBORS, 
        minSize=(minwidth, minheight)
    )
    
    # For every detected eye
    for eye_counter, (x, y, w, h) in enumerate(eyes):
        logging.tell(f'---- Current eye: {eye_counter} ----')

        # Make eye selection a bit smaller
        y = y + defaults.Y_TOP_REDUCTION
        h = h - defaults.Y_BOTTOM_REDUCTION
        x = x + defaults.X_LEFT_OFFSET
        w = w - defaults.X_RIGHT_OFFSET

        # Extract eye from the image
        eye = FACE[y:y+h, x:x+w]

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
            cutoffs.append(
                [
                    x_value, 
                    rel.ellipse_Y(w, h, x_value)
                ]
            )

        # Pixels that do not count
        non_discarded_pixels = 0

        for (cutoff_x, cutoff_y) in cutoffs:
            non_discarded_pixels += 4 * cutoff_y

            cutoff_y_upper = int(h/2) - cutoff_y
            cutoff_y_lower = int(h/2) + cutoff_y
            
            for y_row in range(0, cutoff_y_upper):
                r[y_row][:cutoff_x] = 0
                r[y_row][w - cutoff_x:] = 0
                
            for y_row in range(cutoff_y_lower, h):
                r[y_row][:cutoff_x] = 0
                r[y_row][w - cutoff_x:] = 0

        logging.tell(f'\ttotal pixels: {h * w}p')
        logging.tell(f'\tnon-discarded pixels: {non_discarded_pixels}p')
        logging.tell(f'\tpercentage of discarded pixels: {np.round(1 - (non_discarded_pixels / (h * w)), 3)}%')

        # Filter to detect redness
        f = eye[:, :, 0].astype(np.float)

        # https://ieeexplore.ieee.org/document/1038147
        for row, (blue_row, green_row, red_row) in enumerate(zip(b, g, r)):
            for column, (blue, green, red) in enumerate(zip(blue_row, green_row, red_row)):
                f[row, column] = float(
                    math.pow(red, 2) / (math.pow(blue, 2) + math.pow(green, 2) + 1)
                )

        # Add the green and blue channels together.
        bg = cv2.add(b, g)

        # Generate the redness mask.
        # mask = (r > 80) &  (r > bg)
        mask = (f > defaults.F_VALUE) &\
               (r > (defaults.RB_TEMP * b)) &\
               (r > (defaults.RG_TEMP * g)) &\
               (r > defaults.R_MINIMUM)
        
        # Convert the mask to uint8 format.
        mask = mask.astype(np.uint8)*255

        # Pixels that meet the requirement.
        masked_pixels = 0
        for row in mask:
            unique, counts = np.unique(row, return_counts=True)
            unique_dict = dict(zip(unique, counts))
            try:
                masked_pixels += unique_dict[255]
            except KeyError:
                masked_pixels += 0

        logging.tell(f'\tmasked pixels: {masked_pixels}p')
        logging.tell(f'\tmasked percentage: {np.round(masked_pixels / non_discarded_pixels, 3)}%')

        # Clean mask -- 1) Fill holes 2) Dilate (expand) mask.
        # mask = rel.fillHoles(mask)
        # mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

        # Calculate the mean channel by averaging the green and blue channels.
        mean = bg / 2
        mean = mean[:, :, np.newaxis]
        mask = mask.astype(np.bool)[:, :, np.newaxis]

        # Copy the eye from the original image.
        eyeOut = eye.copy()

        # Copy the mean image to the output image.
        # np.copyto(eyeOut, mean, where=mask)
        eyeOut_higher = np.where(mask, mean, eyeOut)

        # Copy the fixed eye to the output image.
        FACE_OUT[y:y+h, x:x+w, :] = eyeOut_higher

    # Display Result
    cv2.imshow('Red Eyes', FACE)
    cv2.imshow('Red Eyes Removed', FACE_OUT)
    cv2.waitKey(0)
