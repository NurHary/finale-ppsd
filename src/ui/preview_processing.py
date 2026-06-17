import sys
import os

import dearpygui.dearpygui as dpg
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import mpp
from preprocessing.streamer import VideoStreamer
import vidpath


G_enable_Detect = True
G_enable_Crop = True
G_enable_Align = True
G_enable_Landmarks = True

player = VideoStreamer()


def loadVideo(sender, data):
    if not data:
        return
    path = data["file_path_name"]
    player.open(path)


def on_play():
    if player.isfinished:
        player.resetStream()
    player.isstreaming = not player.isstreaming


def on_stop():
    player.resetStream()


def onEnableDetect(sender):
    G_enable_Detect = dpg.get_value(sender)


def onEnableCrop(sender):
    G_enable_Crop = dpg.get_value(sender)


def onEnableAllignment(sender):
    G_enable_Align = dpg.get_value(sender)


def onEnableLandmarks(sender):
    G_enable_Landmarks = dpg.get_value(sender)


def _renderImage(
    frame,
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
        cropframe.flatten(),
        alignframe.flatten(),
        landmarksframe.flatten(),
        voidlandmarksframe.flatten(),
        righteyeframe.flatten(),
        lefteyeframe.flatten(),
        noseframe.flatten(),
    )


def launchUi():
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
    with dpg.window(label="Original Streams", width=640, height=620):
        dpg.add_button(label="Explore Videos", callback=lambda: dpg.show_item("picker"))
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Play",
                callback=on_play,
            )

            dpg.add_button(
                label="Stop",
                callback=on_stop,
            )

        dpg.add_separator()
        dpg.add_text(
            "",
            tag="status",
        )
        dpg.add_image("ov")

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
        player.stream()
        player.checkTimeStamp()

        (tex, crtex, altex, landtex, voidlandtex, rectex, lectex, nctex) = _renderImage(
            player.frame,
            player.cropframe,
            player.alignframe,
            player.landmarksframe,
            player.voidlandmarksframe,
            player.righteyeframe,
            player.lefteyeframe,
            player.noseframe,
        )
        dpg.set_value("ov", tex)
        status = "PLAYING" if player.isstreaming else "PAUSED"

        if player.isfinished:
            status = "FINISHED"
        dpg.set_value(
            "status",
            (f"{status} | {player.curptrstream}/{player.videolentotal}"),
        )
        dpg.render_dearpygui_frame()
        if dpg.is_key_pressed(dpg.mvKey_Spacebar):
            on_play()
    dpg.destroy_context()


def main():
    launchUi()


if __name__ == "__main__":
    main()
