# Name: vidpath.py
# Desc: File dengan variable global constant yang menyimpan strings path

import os
import pathlib

# Parent Path
GC_PATH = pathlib.Path(__file__).parent.parent.parent.resolve()

GC_IMGPATH = os.path.join(GC_PATH, "img/")

# Csv Path
GC_CSV = os.path.join(GC_PATH, "img/csv")

# Real Image Path
GC_CLEARVIDPATH = os.path.join(GC_PATH, "img/original")

# Fake Image Path
GC_NEURALVIDPATH = os.path.join(GC_PATH, "img/NeuralTextures")
GC_FFVIDPATH = os.path.join(GC_PATH, "img/Face2Face")
GC_FACESHIFTVIDPATH = os.path.join(GC_PATH, "img/FaceShifter")
GC_FACESWAPVIDPATH = os.path.join(GC_PATH, "img/FaceSwap")
GC_DFVIDPATH = os.path.join(GC_PATH, "img/DeepFakes")
GC_DFDVIDPATH = os.path.join(GC_PATH, "img/DeepFakesDetection")

# Model Path
GC_MODELPATH = os.path.join(GC_PATH, "models")

L_EYE_SUBGROUP = [33, 133, 160, 159, 158, 157, 173]

R_EYE_SUBGROUP = [263, 362, 387, 386, 385, 384, 398]

NOSE_SUBGROUP = [168, 6, 197, 195, 4]


if __name__ == "__main__":
    pass
