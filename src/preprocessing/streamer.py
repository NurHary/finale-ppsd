import numpy as np
import cv2
from numpy._core.numeric import dtype, uint8

from . import mpp
from collections import deque


class VideoStreamer:
    def __init__(self):
        self.cap = None
        self.path = None

        # PLAYBACK
        self.isstreaming = False
        self.isfinished = False

        self.videolentotal = 0
        self.curptrstream = 0
        self.curmptimestamp = 0
        self.intervaltime = 5

        self.mpmodels = mpp.MedPipeModels()

        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cropframe = np.zeros((480, 640, 3), dtype=np.uint8)
        self.alignframe = np.zeros((256, 256, 3), dtype=np.uint8)
        self.landmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)
        self.voidlandmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)

        # Sub Gorup
        self.lefteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
        self.righteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
        self.noseframe = np.zeros((128, 128, 3), dtype=np.uint8)

        self.history = deque(maxlen=30)
        self.features = {}
        self.batchcache = []

    # # STREAMER UTILS # #
    def open(self, path):
        self.history.clear()
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.videolentotal = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.curptrstream = 0
        self.isfinished = False
        self.isstreaming = False

    def startStream(self):
        if self.isfinished:
            self.resetStream()
        self.isstreaming = True

    def resetStream(self):
        if self.cap:
            self.cap.release()
        self.isfinished = False
        self.isstreaming = False
        self.curptrstream = 0

    def framesampling(self):
        "Fungsi untuk melakukan framesampling"
        return self.curptrstream % self.intervaltime == 0

    def resetAll(self):
        """Fungsi untuk reset model serta timestamp model.
        gunakan ini untuk menghindari int overflow"""
        self.curmptimestamp = 0
        self.mpmodels.resetModel()

    def checkTimeStamp(self):
        "Fungsi untuk mengecek dan reset timestamp"
        if self.curmptimestamp > 3600000:
            self.resetAll()

    def updateTemporalBuffer(self):
        "Fungsi untuk menambahkan nilai saat ini ke t buff"
        if (
            not self.mpmodels.reslandmark2
            or not self.mpmodels.reslandmark2.face_landmarks
        ):
            return

        self.history.append(
            {
                "ts": self.curmptimestamp,
                "landmark": self.mpmodels.reslandmark2.face_landmarks[0],
                "left_eye": self.lefteyeframe,
                "right_eye": self.righteyeframe,
                "nose": self.noseframe,
                "feature": self.extractFeatures(),
            }
        )

    def getBatch(self, size=16):
        if len(self.history) < size:
            return None
        return list(self.history)[-size:]

    # # FACES UTILS # #
    def cropFaces(self):
        """Fungsi untuk mengcrop bagian wajah saja
        dengan nilai dari face detector"""
        if len(self.frame) == 0:
            self.cropframe = np.zeros((256, 256, 3), dtype=np.uint8)
            return False
        if len(self.mpmodels.faces) == 0:
            self.cropframe = np.zeros((256, 256, 3), dtype=np.uint8)
            return False

        x, y, w, h = self.mpmodels.faces[0]
        pad = int(max(w, h) * 0.2)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(self.frame.shape[1], x + w + pad)
        y1 = min(self.frame.shape[0], y + h + pad)

        crop = self.frame[y0:y1, x0:x1]
        if crop.size == 0:
            self.cropframe = np.zeros((256, 256, 3), dtype=np.uint8)
            return False
        self.cropframe = cv2.resize(crop, (256, 256))
        return True

    def alignFaces(self):
        """Fungsi untuk melakukan rotasi pada wajah
        sesuai dengan landmark model 1"""

        if (
            not self.mpmodels.reslandmark1
            or not self.mpmodels.reslandmark1.face_landmarks
        ):
            self.alignframe = np.zeros((256, 256, 3), dtype=uint8)
            return

        h, w = self.cropframe.shape[:2]
        left = self.mpmodels.reslandmark1.face_landmarks[0][33]
        right = self.mpmodels.reslandmark1.face_landmarks[0][263]
        lx = int(left.x * w)
        ly = int(left.y * h)
        rx = int(right.x * w)
        ry = int(right.y * h)
        angle = np.degrees(np.arctan2(ry - ly, rx - lx))

        center = ((lx + rx) // 2, (ly + ry) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(self.cropframe, M, (w, h))

        self.alignframe = aligned

    # # FEATURE EXTRACTOR # #

    def extractEAR(self, idx):
        if (
            not self.mpmodels.reslandmark2
            or not self.mpmodels.reslandmark2.face_landmarks
        ):
            return 0.0
        lm = self.mpmodels.reslandmark2.face_landmarks[0]
        p = []
        for i in idx:
            p.append(np.array([lm[i].x, lm[i].y]))
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        if C == 0:
            return 0
        return (A + B) / (2 * C)

    def extractPose(self):
        if (
            not self.mpmodels.reslandmark2
            or not self.mpmodels.reslandmark2.face_landmarks
        ):
            return 0
        lm = self.mpmodels.reslandmark2.face_landmarks[0]
        nose = lm[4]
        left = lm[234]
        right = lm[454]
        center = (left.x + right.x) / 2
        return nose.x - center

    def extractTexture(self, img):
        if img.size == 0:
            return 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.var(gray)

    def extractFeatures(self):
        left = self.extractEAR(mpp.GC_L_EYE_SUBGROUP)
        right = self.extractEAR(mpp.GC_R_EYE_SUBGROUP)

        pose = self.extractPose()

        tex_eye = self.extractTexture(self.lefteyeframe)
        tex_nose = self.extractTexture(self.noseframe)

        self.features = {
            "ear_left": left,
            "ear_right": right,
            "yaw": pose,
            "eye_texture": tex_eye,
            "nose_texture": tex_nose,
        }

        return self.features

    def getModelInput(self, seq_len=16):
        batch = self.getBatch(seq_len)
        if batch is None:
            return None
        eye = []
        nose = []
        feat = []
        for f in batch:
            eye_img = f["left_eye"].astype(np.float32) / 255.0
            nose_img = f["nose"].astype(np.float32) / 255.0
            eye.append(eye_img)
            nose.append(nose_img)
            feat.append(
                [
                    f["feature"]["ear_left"],
                    f["feature"]["ear_right"],
                    f["feature"]["yaw"],
                    f["feature"]["eye_texture"],
                    f["feature"]["nose_texture"],
                ]
            )

        eye = np.array(eye)
        nose = np.array(nose)
        feat = np.array(feat, dtype=np.float32)
        return {"eye": eye, "nose": nose, "feature": feat}

    # # STREAM MANIPULATION OUTPUT # #
    def drawLandmarks(self, b=None):
        if (
            not self.mpmodels.reslandmark2
            or not self.mpmodels.reslandmark2.face_landmarks
        ):
            if b:
                self.landmarksframe = np.zeros((256, 256, 3), dtype=uint8)
            else:
                self.voidlandmarksframe = np.zeros((256, 256, 3), dtype=uint8)
            return
        if b:
            out = b.copy()
            h, w = self.cropframe.shape[:2]
        else:
            h, w = 256, 256
            out = np.zeros((h, w, 3), dtype=np.uint8)
        for p in self.mpmodels.reslandmark2.face_landmarks[0]:
            x = int(p.x * w)
            y = int(p.y * h)
            cv2.circle(out, (x, y), 2, (255, 0, 255), -1)

        if b:
            self.landmarksframe = out
        else:
            self.voidlandmarksframe = out

    def cropSubGroup(self):
        if (
            not self.mpmodels.reslandmark2
            or not hasattr(self.mpmodels.reslandmark2, "face_landmarks")
            or len(self.mpmodels.reslandmark2.face_landmarks) == 0
        ):
            self.lefteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
            self.righteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
            self.noseframe = np.zeros((128, 128, 3), dtype=np.uint8)
            return

        h, w = self.alignframe.shape[:2]
        pts = []
        for idx in mpp.GC_L_EYE_SUBGROUP:
            p = self.mpmodels.reslandmark2.face_landmarks[0][idx]
            pts.append((int(p.x * w), int(p.y * h)))
        pts = np.array(pts)
        x1 = max(0, pts[:, 0].min() - 20)
        y1 = max(0, pts[:, 1].min() - 20)
        x2 = min(w, pts[:, 0].max() + 20)
        y2 = min(h, pts[:, 1].max() + 20)
        crop = self.alignframe[y1:y2, x1:x2]
        if crop.size == 0:
            self.lefteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
            return
        self.lefteyeframe = cv2.resize(crop, [128, 128])

        pts = []
        for idx in mpp.GC_R_EYE_SUBGROUP:
            p = self.mpmodels.reslandmark2.face_landmarks[0][idx]
            pts.append((int(p.x * w), int(p.y * h)))
        pts = np.array(pts)
        x1 = max(0, pts[:, 0].min() - 20)
        y1 = max(0, pts[:, 1].min() - 20)
        x2 = min(w, pts[:, 0].max() + 20)
        y2 = min(h, pts[:, 1].max() + 20)
        crop = self.alignframe[y1:y2, x1:x2]
        if crop.size == 0:
            self.righteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
            return
        self.righteyeframe = cv2.resize(crop, [128, 128])

        pts = []
        for idx in mpp.GC_NOSE_SUBGROUP:
            p = self.mpmodels.reslandmark2.face_landmarks[0][idx]
            pts.append((int(p.x * w), int(p.y * h)))
        pts = np.array(pts)
        x1 = max(0, pts[:, 0].min() - 20)
        y1 = max(0, pts[:, 1].min() - 20)
        x2 = min(w, pts[:, 0].max() + 20)
        y2 = min(h, pts[:, 1].max() + 20)
        crop = self.alignframe[y1:y2, x1:x2]
        if crop.size == 0:
            self.noseframe = np.zeros((128, 128, 3), dtype=np.uint8)
            return
        self.noseframe = cv2.resize(crop, [128, 128])

    # # STREAM OUT # #
    def stream(self):
        if not self.cap or not self.isstreaming or self.isfinished:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.isfinished = True
            self.isstreaming = False
            return
        self.curptrstream += 1
        detframe = frame.copy()
        self.frame = frame
        if self.framesampling():
            self.mpmodels.detectFaces(detframe, self.curmptimestamp)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            self.curmptimestamp += int(1000 / fps)
            self.cropFaces()
            self.mpmodels.makeLandmarks(self.cropframe, self.curmptimestamp, 0)
            self.alignFaces()
            self.mpmodels.makeLandmarks(self.alignframe, self.curmptimestamp, 1)
            self.drawLandmarks()
            self.cropSubGroup()
            self.updateTemporalBuffer()
