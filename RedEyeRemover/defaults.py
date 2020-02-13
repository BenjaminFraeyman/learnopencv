from pathlib import Path

# Console output of values
VERBOSE = True

# Save console output to file
LOG_SAVE = False
LOG_SAVE_DESTINATION = Path("RedEyeRemover/logs/logfile.txt")

# Input image
# INPUT_IMAGE = Path("RedEyeRemover/Pictures/Bloodshot/edited2.jpg")
# INPUT_IMAGE = Path("RedEyeRemover/Pictures/Bloodshot/irritated.png")
INPUT_IMAGE = Path("RedEyeRemover/Pictures/Colourtarget/face_with_target2.png")

# Colourtarget
TARGET_IMAGE = Path("RedEyeRemover/Pictures/Colourtarget/target_round2.png")

# Selected cascade classifier
CLASSIFIER_CASCADE = Path("RedEyeRemover/HaarCascades/test1.xml")
# CLASSIFIER_CASCADE = Path("RedEyeRemover/HaarCascades/default.xml")

# Cascade parameters
SCALEFACTOR = 1.05
MIN_NEIGHBORS = 8

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

# Filter values
F_VALUE = 0.8
RB_TEMP = 1.0
RG_TEMP = 1.0
R_MINIMUM = 100