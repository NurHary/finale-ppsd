# Name: preprocessing.py
# Desc: File dengan beragam fungsi untuk melakukan preprocessing pada image serta menyediakan stream videos

# stdpy
import argparse
import os

# local files
import vidpath

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

        self.fps = 0
        self.detect_ts = 0
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detectframe = np.zeros((480, 640, 3), dtype=np.uint8)
        self.cropframe = np.zeros((480, 640, 3), dtype=np.uint8)
        self.alignframe = np.zeros((256, 256, 3), dtype=np.uint8)

    def open(self, path):
        if self.detect_ts != 0:
            self.stop()
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

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
            self.frame = frame
            self.detectframe = detframe
        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            0,
        )

    def toggle(self):
        if self.finished:
            self.stop()
            self.toggle()
        else:
            self.playing = not self.playing

    def stop(self):
        self.detect_ts = 0
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
            self.frame = img
            self.detectframe = img

        self.detect_ts = 0
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

        detframe = frame.copy()
        if temporalPartition(self.current, 3):
            faces = detectFaces(frame, self.detect_ts)
            if self.enable_detect:
                self.detect_ts = int(self.current / self.fps) * 1000
                detframe = drawFaces(frame, faces)
            if self.enable_crop and self.enable_allignment:
                self.cropframe = cropFaces(frame, faces)
                result = makeLandmarksFaces(self.cropframe, self.detect_ts)

                if result and result.face_landmarks:
                    self.alignframe = rotateFaces(
                        self.cropframe, result.face_landmarks[0]
                    )

                else:
                    self.alignframe = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                self.cropframe = np.zeros((256, 256, 3), dtype=np.uint8)
                self.alignframe = np.zeros((256, 256, 3), dtype=np.uint8)

        self.detectframe = detframe
        self.frame = frame

    def resetDetector(self):
        self.detect_ts = 0


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
face_landmarker = vision.FaceLandmarker.create_from_options(_LANDMARKS_OPT)


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


def makeLandmarksFaces(f, ts):
    rgb = cv2.cvtColor(
        f,
        cv2.COLOR_BGR2RGB,
    )

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    try:
        return face_landmarker.detect_for_video(
            mp_img,
            ts,
        )

    except:
        return None


def drawFaces(frame, faces):
    out = frame.copy()
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


# Fungsi untuk mengambil video dari path secara berkala, selain itu juga fungsi ini akan memanggil fungsi detectFaces
# untuk mendeteksi wajah dan melakukan crop pada area dekat wajah itu
def videoStreamOut(p: str, i: int, ti: str):
    cap = cv2.VideoCapture(p)

    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {p}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if temporalPartition(frame_idx, i):
            yield frame_idx, frame
        frame_idx += 1

    cap.release()


def _renderFrame(frame, detframe, cropframe, alignframe):
    frame = cv2.resize(frame, (640, 480))
    detframe = cv2.resize(detframe, (640, 480))
    cropframe = cv2.resize(cropframe, (256, 256))
    alignframe = cv2.resize(alignframe, (256, 256))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detframe = cv2.cvtColor(detframe, cv2.COLOR_BGR2RGB)
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2RGB)
    alignframe = cv2.cvtColor(alignframe, cv2.COLOR_BGR2RGB)

    frame = frame.astype(np.float32) / 255.0
    detframe = detframe.astype(np.float32) / 255.0
    cropframe = cropframe.astype(np.float32) / 255.0
    alignframe = alignframe.astype(np.float32) / 255.0

    return (
        frame.flatten(),
        detframe.flatten(),
        cropframe.flatten(),
        alignframe.flatten(),
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

        (tex, dettex, crtex, altex) = _renderFrame(
            player.frame, player.detectframe, player.cropframe, player.alignframe
        )

        dpg.set_value("ov", tex)
        dpg.set_value("fdv", dettex)
        dpg.set_value("cvv", crtex)
        dpg.set_value("av", altex)

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
