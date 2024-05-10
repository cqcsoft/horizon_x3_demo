from hobot_vio import libsrcampy as srcampy
import numpy as np
import cv2
import os


class Displayer(object):
    def __init__(self, width=1920, height=1080) -> None:
        # def get_display_res():
        #     res_path = "/usr/bin/get_hdmi_res"
        #     if not os.path.exists(res_path):
        #         return 1920, 1080

        #     import subprocess
        #     p = subprocess.Popen([res_path], stdout=subprocess.PIPE)
        #     result = p.communicate()
        #     res = result[0].split(b',')
        #     res[1] = max(min(int(res[1]), 1920), 0)
        #     res[0] = max(min(int(res[0]), 1080), 0)
        #     return int(res[1]), int(res[0])

        # Get HDMI display object
        self.disp = srcampy.Display()
        # self.disp_w, self.disp_h = get_display_res()
        self.disp_w, self.disp_h = width, height
        self.disp.display(0, self.disp_w, self.disp_h)

    def __del__(self) -> None:
        self.disp.close()

    def show(self, image: np.ndarray) -> None:
        def bgr2nv12(image: np.ndarray):
            height, width, _ = image.shape
            area = height * width
            yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
            y = yuv420p[:area]
            uv_planar = yuv420p[area:].reshape((2, area // 4))
            uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

            nv12 = np.zeros_like(yuv420p)
            nv12[:height * width] = y
            nv12[height * width:] = uv_packed
            return nv12
        height, width, _ = image.shape
        if height!=self.disp_h or width!=self.disp_w:
            image = cv2.resize(image, (self.disp_w,self.disp_h))
        image = bgr2nv12(image)
        self.disp.set_img(image.tobytes())
