from pathlib import Path

# Console output of values
VERBOSE = True

# Save console output to file
LOG_SAVE = False
LOG_SAVE_DESTINATION = Path("RedEyeRemover/logs/logfile.txt")

# Input image
INPUT_IMAGE = Path("RedEyeRemover/Pictures/Bloodshot/edited8.jpg")
# INPUT_IMAGE = Path("RedEyeRemover/Pictures/Bloodshot/irritated.png")

# Selected cascade classifier
CLASSIFIER_CASCADE = Path("RedEyeRemover/HaarCascades/test1.xml")
# CLASSIFIER_CASCADE = Path("RedEyeRemover/HaarCascades/default.xml")

# Eye ratio compared to a full face
EYE_W_RATIO = 1/7
EYE_H_RATIO = 1/9

# Reducing the height of the eye
Y_TOP_REDUCTION = 30
Y_BOTTOM_REDUCTION = 45

# Reducing the width of the eye
X_LEFT_OFFSET = 20
X_RIGHT_OFFSET = 20

# Reduce redness overall
RED_OFFSET = 20

# Line where redness is selected
REDNESS_FILTER = 0.8