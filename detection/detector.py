import torch
import numpy as np
from .utils.datasets import letterbox
from .utils.general import non_max_suppression, scale_coords
from .models.experimental import attempt_load

class Detector:
    def __init__(self, weight_path, img_size=640):

        self._img_size = img_size
        self._device = torch.device("cuda")
        self._half = self._device.type != 'cpu'
        self._model = attempt_load(weight_path, map_location=self._device)
        self._stride = int(self._model.stride.max())
        self._names = self._model.module.names if hasattr(self._model, 'module') else self._model.names
        if torch.cuda.is_available():
            self._model.cuda()
            if self._half:
                self._model.half()

    def input_model(self, img_src):
        imgChangeSize = letterbox(img_src, self._img_size, auto=True, stride=self._stride)[0]
        inputModelIMG = [imgChangeSize]
        inputModelIMG = np.stack(inputModelIMG, 0)
        inputModelIMG = inputModelIMG[:, :, :, ::-1].transpose(0, 3, 1, 2)
        inputModelIMG = np.ascontiguousarray(inputModelIMG)
        inputModelIMG = torch.from_numpy(inputModelIMG)
        inputModelIMG = inputModelIMG.half() if self._half else inputModelIMG.float()
        inputModelIMG = inputModelIMG.to(self._device)
        inputModelIMG /= 255.0  # 0 - 255 to 0.0 - 1.0
        return inputModelIMG

    def predict_image(self, img):
        inputModelIMG = self.input_model(img)
        pred = self._model(inputModelIMG, augment=False)[0]
        pred = non_max_suppression(pred,  classes=None, agnostic=False)
        return pred, inputModelIMG

    def detect_drug(self, imgSrc, thresh=0.33):
        pred, img = self.predict_image(imgSrc)
        Text = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgSrc.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if float(conf) > thresh:
                        line = cls.item(), f'{conf:.2f}', str(int(xyxy[0])), str(int(xyxy[1])), str(int(xyxy[2])), str(
                            int(xyxy[3]))
                        Text.append(line)
            return Text

    def detection_press(self, imgSrc, thresh=0.3):
        pred, img = self.predict_image(imgSrc)
        Text = []

        for i, det in enumerate(pred):  # detections per image
            if len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgSrc.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if  float(conf) > thresh:
                        press_xmin = int(xyxy[0])
                        press_ymin = int(xyxy[1])
                        press_xmax = int(xyxy[2])
                        press_ymax = int(xyxy[3])
                        imgCrop_press = imgSrc[press_ymin:press_ymax, press_xmin:press_xmax]
                        Text.append(imgCrop_press)
        return Text