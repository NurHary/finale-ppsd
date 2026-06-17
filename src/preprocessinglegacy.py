# Name: preprocessing.py
# Desc: File dengan beragam fungsi untuk melakukan preprocessing pada image serta menyediakan stream videos

# stdpy
import argparse
import os

# local files
import vidpath
from preprocessing.mpp import GC_L_EYE_SUBGROUP, GC_NOSE_SUBGROUP, GC_R_EYE_SUBGROUP

# External Lib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import dearpygui.dearpygui as dpg


class VideoPlayer:
    def __init__(self):
        self.cap = None
        self.path = None
        self.playing = False
        self.finished = False
        self.total = 0
        self.current = 0

        self.enable_detect = True
        self.enable_crop = True
        self.enable_allignment = True
        self.enable_landmarks = True

        self.detect_ts = 0
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detectframe = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cropframe = np.zeros((480, 640, 3), dtype=np.uint8)
        self.alignframe = np.zeros((256, 256, 3), dtype=np.uint8)
        self.landmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)
        self.voidlandmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)

        # TODO: Landmarks Group
        # Sub Gorup
        self.lefteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
        self.righteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
        self.noseframe = np.zeros((128, 128, 3), dtype=np.uint8)

    def open(self, path):
        if self.current != 0:
            self.stop()
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            return

        self.path = path
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current = 0
        self.finished = False
        self.playing = False

        ok, frame = self.cap.read()
        detframe = frame.copy()
        if ok:
            self.detect_ts
            if self.enable_detect and temporalPartition(self.current, 3):
                faces = detectFaces(frame, self.detect_ts)
                detframe = drawFaces(frame, faces)
                self.cropframe = cropFaces(frame, faces)
                self.detect_ts += 1
                result = makeLandmarksFaces(
                    self.cropframe, self.detect_ts, face_allandmaker
                )
                if result and result.face_landmarks:
                    self.alignframe = rotateFaces(
                        self.cropframe, result.face_landmarks[0]
                    )
                else:
                    self.alignframe = np.zeros((256, 256, 3), dtype=np.uint8)

                aligntlandsult = makeLandmarksFaces(
                    self.alignframe, self.detect_ts, face_landmaker
                )

                if aligntlandsult and aligntlandsult.face_landmarks:
                    self.landmarksframe = drawLandmarks(
                        self.alignframe, aligntlandsult.face_landmarks[0]
                    )
                    self.voidlandmarksframe = drawLandmarksVoid(
                        aligntlandsult.face_landmarks[0]
                    )
                    self.lefteyeframe = cropTheFuckingLandmarks(
                        self.alignframe,
                        aligntlandsult.face_landmarks[0],
                        GC_L_EYE_SUBGROUP,
                    )

                    self.righteyeframe = cropTheFuckingLandmarks(
                        self.alignframe,
                        aligntlandsult.face_landmarks[0],
                        GC_R_EYE_SUBGROUP,
                    )
                    self.noseframe = cropTheFuckingLandmarks(
                        self.alignframe,
                        aligntlandsult.face_landmarks[0],
                        GC_NOSE_SUBGROUP,
                    )
                else:
                    self.landmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)
                    self.voidlandmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)
                    self.lefteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
                    self.righteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
                    self.noseframe = np.zeros((128, 128, 3), dtype=np.uint8)
            self.frame = frame
            self.detectframe = detframe
        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0,
        )

    def toggle(self):
        if self.finished:
            self.stop()
        self.playing = not self.playing

    def stop(self):
        self.resetDetector()
        self.playing = False
        self.finished = False
        self.current = 0

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def seek(self, frame):
        if not self.cap:
            return
        self.current = frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = self.cap.retrieve()

        if ok:
            faces = detectFaces(img, self.detect_ts)
            self.frame = img
            self.detectframe = drawFaces(img, faces)
            self.cropframe = cropFaces(img, faces)
            self.detect_ts += 1
            result = makeLandmarksFaces(
                self.cropframe, self.detect_ts, face_allandmaker
            )
            if result and result.face_landmarks:
                self.alignframe = rotateFaces(self.cropframe, result.face_landmarks[0])

            else:
                self.alignframe = np.zeros((256, 256, 3), dtype=np.uint8)

            aligntlandsult = makeLandmarksFaces(
                self.alignframe, self.detect_ts, face_landmaker
            )

            if aligntlandsult and aligntlandsult.face_landmarks:
                self.landmarksframe = drawLandmarks(
                    self.alignframe, aligntlandsult.face_landmarks[0]
                )
                self.voidlandmarksframe = drawLandmarksVoid(
                    aligntlandsult.face_landmarks[0]
                )
                self.lefteyeframe = cropTheFuckingLandmarks(
                    self.alignframe,
                    aligntlandsult.face_landmarks[0],
                    GC_L_EYE_SUBGROUP,
                )

                self.righteyeframe = cropTheFuckingLandmarks(
                    self.alignframe,
                    aligntlandsult.face_landmarks[0],
                    GC_R_EYE_SUBGROUP,
                )
                self.noseframe = cropTheFuckingLandmarks(
                    self.alignframe,
                    aligntlandsult.face_landmarks[0],
                    GC_NOSE_SUBGROUP,
                )
            else:
                self.landmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)
                self.voidlandmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)
                self.lefteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
                self.righteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
                self.noseframe = np.zeros((128, 128, 3), dtype=np.uint8)

        self.resetDetector()

    def update(self):
        if not self.cap or not self.playing or self.finished:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.finished = True
            self.playing = False
            return
        self.current += 1

        self.frame = frame
        detframe = frame.copy()
        if temporalPartition(self.current, 3):
            faces = detectFaces(frame, self.detect_ts)
            if self.enable_detect:
                self.detect_ts += int(self.cap.get(cv2.CAP_PROP_FPS))
                self.detectframe = drawFaces(frame, faces)
            if self.enable_crop and self.enable_allignment:
                self.cropframe = cropFaces(frame, faces)
                result = makeLandmarksFaces(
                    self.cropframe, self.detect_ts, face_allandmaker
                )

                if result and result.face_landmarks and self.enable_allignment:
                    self.alignframe = rotateFaces(
                        self.cropframe, result.face_landmarks[0]
                    )

                else:
                    self.alignframe = np.zeros((256, 256, 3), dtype=np.uint8)

                aligntlandsult = makeLandmarksFaces(
                    self.alignframe, self.detect_ts, face_landmaker
                )
                if (
                    aligntlandsult
                    and aligntlandsult.face_landmarks
                    and self.enable_landmarks
                ):
                    self.landmarksframe = drawLandmarks(
                        self.alignframe, aligntlandsult.face_landmarks[0]
                    )
                    self.voidlandmarksframe = drawLandmarksVoid(
                        aligntlandsult.face_landmarks[0]
                    )
                    self.lefteyeframe = cropTheFuckingLandmarks(
                        self.alignframe,
                        aligntlandsult.face_landmarks[0],
                        GC_L_EYE_SUBGROUP,
                    )

                    self.righteyeframe = cropTheFuckingLandmarks(
                        self.alignframe,
                        aligntlandsult.face_landmarks[0],
                        GC_R_EYE_SUBGROUP,
                    )
                    self.noseframe = cropTheFuckingLandmarks(
                        self.alignframe,
                        aligntlandsult.face_landmarks[0],
                        GC_NOSE_SUBGROUP,
                    )
                else:
                    self.landmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)
                    self.voidlandmarksframe = np.zeros((256, 256, 3), dtype=np.uint8)
                    self.lefteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
                    self.righteyeframe = np.zeros((128, 128, 3), dtype=np.uint8)
                    self.noseframe = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                self.cropframe = np.zeros((256, 256, 3), dtype=np.uint8)
                self.alignframe = np.zeros((256, 256, 3), dtype=np.uint8)

    def resetDetector(self):
        # self.detect_ts += 10
        # # global face_detector
        # global face_allandmaker
        # global face_landmaker
        # face_detector.close()
        # face_detector = vision.FaceDetector.create_from_options(_FACE_OPT)
        # face_allandmaker.close()
        # face_allandmaker = vision.FaceLandmarker.create_from_options(_LANDMARKS_OPT)
        # face_landmaker.close()
        # face_landmaker = vision.FaceLandmarker.create_from_options(_LANDMARKS_OPT)
        pass


player = VideoPlayer()


_BASE_OPTIONS_LANDMARKS = python.BaseOptions(
    model_asset_path=(os.path.join(vidpath.GC_MODELPATH, "landmarks.task"))
)
_BASE_OPTIONS_FACED = python.BaseOptions(
    model_asset_path=(os.path.join(vidpath.GC_MODELPATH, "blaze.tflite"))
)

_FACE_OPT = vision.FaceDetectorOptions(
    base_options=_BASE_OPTIONS_FACED,
    running_mode=(vision.RunningMode.VIDEO),
    min_detection_confidence=0.5,
)
_LANDMARKS_OPT = vision.FaceLandmarkerOptions(
    base_options=_BASE_OPTIONS_LANDMARKS,
    num_faces=1,
    output_face_blendshapes=False,
    running_mode=(vision.RunningMode.VIDEO),
    output_facial_transformation_matrixes=True,
)

face_detector = vision.FaceDetector.create_from_options(_FACE_OPT)
face_allandmaker = vision.FaceLandmarker.create_from_options(_LANDMARKS_OPT)
face_landmaker = vision.FaceLandmarker.create_from_options(_LANDMARKS_OPT)


def loadVideo(sender, data):
    if not data:
        return
    path = data["file_path_name"]
    player.open(path)


def onPlay():
    player.toggle()


def onStop():
    player.stop()


def onTimeline(sender):
    if dpg.is_item_active(sender):
        player.seek(int(dpg.get_value(sender)))


def onEnableDetect(sender):
    player.enable_detect = dpg.get_value(sender)


def onEnableCrop(sender):
    player.enable_crop = dpg.get_value(sender)


def onEnableAllignment(sender):
    player.enable_allignment = dpg.get_value(sender)


def onEnableLandmarks(sender):
    player.enable_landmarks = dpg.get_value(sender)


# Fungsi untuk melakukan partisi waktu pada videos
def temporalPartition(f, i):
    return f % i == 0


# Fungsi untuk melakukan deteksi Wajah
def detectFaces(f, t_ms):
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    mp_img = mp.Image(
        image_format=(mp.ImageFormat.SRGB),
        data=rgb,
    )

    try:
        result = face_detector.detect_for_video(mp_img, t_ms)

    except ValueError:
        return []

    faces = []

    h, w = f.shape[:2]

    for det in result.detections:
        box = det.bounding_box

        x = box.origin_x
        y = box.origin_y

        bw = box.width
        bh = box.height

        faces.append((x, y, bw, bh))

    return faces


def makeLandmarksFaces(f, t_ms, ln):
    rgb = cv2.cvtColor(
        f,
        cv2.COLOR_BGR2RGB,
    )

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    try:
        return ln.detect_for_video(mp_img, t_ms)
    except ValueError:
        return []


def drawFaces(fr, faces):
    out = fr.copy()
    for x, y, w, h in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return out


def drawLandmarks(fr, ln):
    out = fr.copy()
    h, w = fr.shape[:2]
    for p in ln:
        x = int(p.x * w)
        y = int(p.y * h)

        cv2.circle(out, (x, y), 2, (0, 255, 255), -1)

    return out


def drawLandmarksVoid(ln, size=(256, 256)):
    h, w = size
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if ln is None:
        return out
    for p in ln:
        x = int(p.x * w)
        y = int(p.y * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), 2, (0, 255, 255), -1)

    return out


def cropFaces(fr, fa):
    if len(fa) == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    x, y, w, h = fa[0]
    pad = int(max(w, h) * 0.2)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(fr.shape[1], x + w + pad)
    y1 = min(fr.shape[0], y + h + pad)

    crop = fr[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    return cv2.resize(crop, (256, 256))


def cropTheFuckingLandmarks(fr, ln, ids):
    h, w = fr.shape[:2]
    pts = []
    for idx in ids:
        p = ln[idx]
        pts.append((int(p.x * w), int(p.y * h)))

    pts = np.array(pts)
    x1 = max(0, pts[:, 0].min() - 20)
    y1 = max(0, pts[:, 1].min() - 20)
    x2 = min(w, pts[:, 0].max() + 20)
    y2 = min(h, pts[:, 1].max() + 20)
    crop = fr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    return cv2.resize(crop, [128, 128])


def rotateFaces(fr, ln):
    h, w = fr.shape[:2]
    left = ln[33]
    right = ln[263]
    lx = int(left.x * w)
    ly = int(left.y * h)
    rx = int(right.x * w)
    ry = int(right.y * h)
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    center = ((lx + rx) // 2, (ly + ry) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    aligned = cv2.warpAffine(fr, M, (w, h))

    return aligned


def _renderFrame(
    frame,
    detframe,
    cropframe,
    alignframe,
    landmarksframe,
    voidlandmarksframe,
    righteyeframe,
    lefteyeframe,
    noseframe,
):
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0

    detframe = cv2.resize(detframe, (640, 480))
    detframe = cv2.cvtColor(detframe, cv2.COLOR_BGR2RGB)
    detframe = detframe.astype(np.float32) / 255.0

    cropframe = cv2.resize(cropframe, (256, 256))
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2RGB)
    cropframe = cropframe.astype(np.float32) / 255.0

    alignframe = cv2.resize(alignframe, (256, 256))
    alignframe = cv2.cvtColor(alignframe, cv2.COLOR_BGR2RGB)
    alignframe = alignframe.astype(np.float32) / 255.0

    landmarksframe = cv2.resize(landmarksframe, (256, 256))
    landmarksframe = cv2.cvtColor(landmarksframe, cv2.COLOR_BGR2RGB)
    landmarksframe = landmarksframe.astype(np.float32) / 255.0

    voidlandmarksframe = cv2.resize(voidlandmarksframe, (256, 256))
    voidlandmarksframe = cv2.cvtColor(voidlandmarksframe, cv2.COLOR_BGR2RGB)
    voidlandmarksframe = voidlandmarksframe.astype(np.float32) / 255.0

    righteyeframe = cv2.resize(righteyeframe, (128, 128))
    righteyeframe = cv2.cvtColor(righteyeframe, cv2.COLOR_BGR2RGB)
    righteyeframe = righteyeframe.astype(np.float32) / 255.0

    lefteyeframe = cv2.resize(lefteyeframe, (128, 128))
    lefteyeframe = cv2.cvtColor(lefteyeframe, cv2.COLOR_BGR2RGB)
    lefteyeframe = lefteyeframe.astype(np.float32) / 255.0

    noseframe = cv2.resize(noseframe, (128, 128))
    noseframe = cv2.cvtColor(noseframe, cv2.COLOR_BGR2RGB)
    noseframe = noseframe.astype(np.float32) / 255.0

    return (
        frame.flatten(),
        detframe.flatten(),
        cropframe.flatten(),
        alignframe.flatten(),
        landmarksframe.flatten(),
        voidlandmarksframe.flatten(),
        righteyeframe.flatten(),
        lefteyeframe.flatten(),
        noseframe.flatten(),
    )


def _launcUi():
    dpg.create_context()

    with dpg.texture_registry():
        dpg.add_raw_texture(
            640,
            480,
            np.zeros(
                640 * 480 * 3,
                dtype=np.float32,
            ),
            tag="ov",
            format=dpg.mvFormat_Float_rgb,
        )
        dpg.add_raw_texture(
            640,
            480,
            np.zeros(
                640 * 480 * 3,
                dtype=np.float32,
            ),
            tag="fdv",
            format=dpg.mvFormat_Float_rgb,
        )
        dpg.add_raw_texture(
            256,
            256,
            np.zeros(
                256 * 256 * 3,
                dtype=np.float32,
            ),
            tag="cvv",
            format=dpg.mvFormat_Float_rgb,
        )
        dpg.add_raw_texture(
            256,
            256,
            np.zeros(
                256 * 256 * 3,
                dtype=np.float32,
            ),
            tag="av",
            format=dpg.mvFormat_Float_rgb,
        )
        dpg.add_raw_texture(
            256,
            256,
            np.zeros(
                256 * 256 * 3,
                dtype=np.float32,
            ),
            tag="lv",
            format=dpg.mvFormat_Float_rgb,
        )
        dpg.add_raw_texture(
            256,
            256,
            np.zeros(
                256 * 256 * 3,
                dtype=np.float32,
            ),
            tag="vlv",
            format=dpg.mvFormat_Float_rgb,
        )
        dpg.add_raw_texture(
            128,
            128,
            np.zeros(
                128 * 128 * 3,
                dtype=np.float32,
            ),
            tag="lecv",
            format=dpg.mvFormat_Float_rgb,
        )
        dpg.add_raw_texture(
            128,
            128,
            np.zeros(
                128 * 128 * 3,
                dtype=np.float32,
            ),
            tag="recv",
            format=dpg.mvFormat_Float_rgb,
        )
        dpg.add_raw_texture(
            128,
            128,
            np.zeros(
                128 * 128 * 3,
                dtype=np.float32,
            ),
            tag="ncv",
            format=dpg.mvFormat_Float_rgb,
        )

    with dpg.window(
        label="Original Streams",
        width=640,
        height=620,
    ):
        dpg.add_button(
            label="Open Video",
            callback=lambda: dpg.show_item("picker"),
        )

        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Play",
                callback=onPlay,
            )

            dpg.add_button(
                label="Stop",
                callback=onStop,
            )

        dpg.add_separator()

        dpg.add_slider_int(
            label="Timeline",
            tag="timeline",
            min_value=0,
            max_value=1,
            callback=onTimeline,
        )

        dpg.add_text(
            "",
            tag="status",
        )

        dpg.add_image("ov")

    with dpg.window(label="Face Detector", width=640, height=480, pos=[640, 0]):
        dpg.add_image("fdv")
    with dpg.window(label="Face Crops", width=256, height=256, pos=[1280, 0]):
        dpg.add_image("cvv")
    with dpg.window(label="Face Allignment", width=256, height=256, pos=[1280, 256]):
        dpg.add_image("av")
    with dpg.window(label="Face Landmarks", width=256, height=256, pos=[1536, 0]):
        dpg.add_image("lv")
    with dpg.window(
        label="Void Face Landmarks", width=256, height=256, pos=[1536, 256]
    ):
        dpg.add_image("vlv")
    with dpg.window(label="Right Eye Crop", width=128, height=128, pos=[1536, 512]):
        dpg.add_image("recv")
    with dpg.window(
        label="Left Eye Crop",
        width=128,
        height=128,
        pos=[1280, 512],  # 1536
    ):
        dpg.add_image("lecv")
    with dpg.window(label="Nose Crop", width=128, height=128, pos=[1408, 512]):
        dpg.add_image("ncv")

    with dpg.window(
        label="Control Panels",
        width=640,
        height=140,
        pos=[640, 480],
        max_size=[3000, 140],
    ):
        dpg.add_checkbox(
            label="Face Detect", default_value=True, callback=onEnableDetect
        )
        dpg.add_checkbox(label="Face Crops", default_value=True, callback=onEnableCrop)
        dpg.add_checkbox(
            label="Face Allignment", default_value=True, callback=onEnableAllignment
        )
        dpg.add_checkbox(
            label="Face Landmarks", default_value=True, callback=onEnableLandmarks
        )

    with dpg.file_dialog(
        label="File Selector",
        height=400,
        directory_selector=False,
        show=False,
        default_path=vidpath.GC_IMGPATH,
        callback=loadVideo,
        tag="picker",
        modal=True,
    ):
        dpg.add_file_extension(".mp4")
        dpg.add_file_extension(".avi")
        dpg.add_file_extension(".mov")

    dpg.create_viewport(
        title="Deepfake Viewer",
        width=1400,
        height=1400,
    )

    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        player.update()

        (tex, dettex, crtex, altex, landtex, voidlandtex, rectex, lectex, nctex) = (
            _renderFrame(
                player.frame,
                player.detectframe,
                player.cropframe,
                player.alignframe,
                player.landmarksframe,
                player.voidlandmarksframe,
                player.righteyeframe,
                player.lefteyeframe,
                player.noseframe,
            )
        )

        dpg.set_value("ov", tex)
        dpg.set_value("fdv", dettex)
        dpg.set_value("cvv", crtex)
        dpg.set_value("av", altex)
        dpg.set_value("lv", landtex)
        dpg.set_value("vlv", voidlandtex)
        dpg.set_value("recv", rectex)
        dpg.set_value("lecv", lectex)
        dpg.set_value("ncv", nctex)

        dpg.configure_item(
            "timeline",
            max_value=max(
                1,
                player.total,
            ),
        )

        dpg.set_value(
            "timeline",
            player.current,
        )

        status = "PLAYING" if player.playing else "PAUSED"

        if player.finished:
            status = "FINISHED"

        dpg.set_value(
            "status",
            (f"{status} | {player.current}/{player.total}"),
        )

        dpg.render_dearpygui_frame()

        if dpg.is_key_pressed(dpg.mvKey_Spacebar):
            player.toggle()
        if dpg.is_key_pressed(dpg.mvKey_Right):
            player.seek(min(player.total, player.current + 50))
        if dpg.is_key_pressed(dpg.mvKey_Left):
            player.seek(max(0, player.current - 50))

        if dpg.is_key_released(dpg.mvKey_D):
            player.enable_detect = not player.enable_detect

    dpg.destroy_context()


def main():
    parser = argparse.ArgumentParser(
        prog="Image Preprocessing Sys",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "-su",
        "--showui",
        help="Argument untuk menunjukkan ui imgui",
        default=False,
        action="store_true",
        required=False,
    )
    args = parser.parse_args()

    if args.showui:
        _launcUi()


if __name__ == "__main__":
    main()
