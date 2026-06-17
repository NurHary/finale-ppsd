# Name: mpp.py
# Desc: Mediapipe Preprocessor moduls, menyediakan beberapa config opt untuk mediapipe dan

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

import os
import vidpath

GC_L_EYE_SUBGROUP = [33, 133, 160, 159, 158, 157, 173]
GC_R_EYE_SUBGROUP = [263, 362, 387, 386, 385, 384, 398]
GC_NOSE_SUBGROUP = [168, 6, 197, 195, 4]


# CONFIG GLOBAL CONSTANT
GC_BASEOPT_LM = python.BaseOptions(
    model_asset_path=os.path.join(vidpath.GC_MODELPATH, "landmarks.task")
)

GC_BASEOPT_FD = python.BaseOptions(
    model_asset_path=os.path.join(vidpath.GC_MODELPATH, "blaze.tflite")
)

GC_FDOPT = vision.FaceDetectorOptions(
    base_options=GC_BASEOPT_FD,
    running_mode=(vision.RunningMode.VIDEO),
    min_detection_confidence=0.5,
)

GC_LMOPT = vision.FaceLandmarkerOptions(
    base_options=GC_BASEOPT_LM,
    num_faces=1,
    output_face_blendshapes=False,
    running_mode=(vision.RunningMode.VIDEO),
    output_facial_transformation_matrixes=True,
)


class MedPipeModels:
    def __init__(self):
        self.facedetm1 = vision.FaceDetector.create_from_options(GC_FDOPT)
        self.landmarkm1 = vision.FaceLandmarker.create_from_options(GC_LMOPT)
        self.landmarkm2 = vision.FaceLandmarker.create_from_options(GC_LMOPT)

        self.faces = []
        self.reslandmark1 = []
        self.reslandmark2 = []

    def resetModel(self):
        "Fungsi untuk mereset dan init ulang sesuai opt"
        self.facedetm1.close()
        self.landmarkm1.close()
        self.landmarkm2.close()
        self.facedetm1 = vision.FaceDetector.create_from_options(GC_FDOPT)
        self.landmarkm1 = vision.FaceLandmarker.create_from_options(GC_LMOPT)
        self.landmarkm2 = vision.FaceLandmarker.create_from_options(GC_LMOPT)

    def detectFaces(self, f, tms):
        """Fungsi untuk mendeteksi faces"""
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(image_format=(mp.ImageFormat.SRGB), data=rgb)

        try:
            result = self.facedetm1.detect_for_video(mp_img, tms)
        except ValueError:
            return []

        faces = []
        for det in result.detections:
            box = det.bounding_box

            x = box.origin_x
            y = box.origin_y

            bw = box.width
            bh = box.height

            faces.append((x, y, bw, bh))

        self.faces = faces

    def makeLandmarks(self, f, tms, n):
        """Fungsi untuk membuat landmarks dengan input tambahan berupa n,
        apabila n = 0 maka landmarkm1 dan apabila n != 0 maka landmarkm2"""
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            if n == 0:
                self.reslandmark1 = self.landmarkm1.detect_for_video(mp_img, tms)
            else:
                self.reslandmark2 = self.landmarkm2.detect_for_video(mp_img, tms)
        except ValueError:
            if n == 0:
                self.reslandmark1 = []
            else:
                self.reslandmark2 = []
