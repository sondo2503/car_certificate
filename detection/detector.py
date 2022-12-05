import queue

import torch
import numpy as np
import cv2
import os
from .utils.general import non_max_suppression, scale_coords
from src.tools.croper import Croper
from .models.experimental import attempt_load

class Detector:
    def __init__(self, weight_path, img_size=640):

        self._img_size = img_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._half = self._device.type != 'cpu'
        self._model = attempt_load(weight_path, map_location=self._device)
        self._stride = int(self._model.stride.max())
        self._names = self._model.module.names if hasattr(self._model, 'module') else self._model.names
        self.croper = Croper()
        if torch.cuda.is_available():
            self._model.cuda()
            if self._half:
                self._model.half()

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def strip_optimizer(f='best_last.pt', s=''):  # from utils.general import *; strip_optimizer()
        # Strip optimizer from 'f' to finalize training, optionally save as 's'
        x = torch.load(f, map_location=torch.device('cpu'))
        if x.get('ema'):
            x['model'] = x['ema']  # replace model with ema
        for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
            x[k] = None
        x['epoch'] = -1
        x['model'].half()  # to FP16
        for p in x['model'].parameters():
            p.requires_grad = False
        torch.save(x, s or f)
        mb = os.path.getsize(s or f) / 1E6  # filesize
        print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")

    def makeInputModel(self, img_src):
        inputModelIMG = self.letterbox(img_src, self._img_size, auto=True, stride=self._stride)[0]
        img = inputModelIMG[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict_IMG(self, img):
        inputModelIMG = self.makeInputModel(img)
        pred = self._model(inputModelIMG, augment=False)[0]
        pred = non_max_suppression(pred, 0.4, 0.25, classes=None, agnostic=False)
        return pred, inputModelIMG
    def detect_page(self, imgSrc, thresh=0.5):

        pred, img = self.predict_IMG(imgSrc)

        name_page = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgSrc.shape).round()
                for *xyxy, conf, cls in reversed(det):

                    # if (self._names[int(cls)] == 'page' and float(conf) > thresh):
                    #     page_xmin = int(xyxy[0])
                    #     page_ymin = int(xyxy[1])
                    #     page_xmax = int(xyxy[2])
                    #     page_ymax = int(xyxy[3])
                    #     imgCrop = imgSrc[page_ymin:page_ymax, page_xmin:page_xmax]
                    #     name_page.append(((self._names[int(cls)]), imgCrop))
                    if (self._names[int(cls)] == 'page_1' and float(conf) > thresh):
                        page_1_xmin = int(xyxy[0])
                        page_1_ymin = int(xyxy[1])
                        page_1_xmax = int(xyxy[2])
                        page_1_ymax = int(xyxy[3])
                        imgCrop = imgSrc[page_1_ymin:page_1_ymax, page_1_xmin:page_1_xmax]
                        name_page.append(((self._names[int(cls)]), imgCrop))
                    if (self._names[int(cls)] == 'page_2' and float(conf) > thresh):
                        page_2_xmin = int(xyxy[0])
                        page_2_ymin = int(xyxy[1])
                        page_2_xmax = int(xyxy[2])
                        page_2_ymax = int(xyxy[3])
                        imgCrop = imgSrc[page_2_ymin:page_2_ymax, page_2_xmin:page_2_xmax]
                        name_page.append(((self._names[int(cls)]), imgCrop))
                    if (self._names[int(cls)] == 'page_3' and float(conf) > thresh):
                        page_1_xmin = int(xyxy[0])
                        page_1_ymin = int(xyxy[1])
                        page_1_xmax = int(xyxy[2])
                        page_1_ymax = int(xyxy[3])
                        imgCrop = imgSrc[page_1_ymin:page_1_ymax, page_1_xmin:page_1_xmax]
                        name_page.append(((self._names[int(cls)]), imgCrop))
        return name_page

    def detect_license_plates_page(self, imgSrc, thresh=0.4):
        pred, img = self.predict_IMG(imgSrc)
        license_plates_A = []


        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # names: [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgSrc.shape).round()
                for *xyxy, conf, cls in reversed(det):

                    if (self._names[int(cls)] == 'A' and float(conf) > thresh):
                        license_plates_A.append(self.croper.cropByCoor(imgSrc,
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   3])),
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   3]))))
        return license_plates_A

    def detect_license_plates2(self, imgSrc, thresh=0.4):
        pred, img = self.predict_IMG(imgSrc)
        license_plates_A = []
        license_plates_B = []
        license_plates_C = []
        license_plates_D = []
        license_plates_E = []
        license_plates_F = []
        license_plates_G = []
        license_plates_H = []
        license_plates_I = []
        license_plates_C1 = []
        license_plates_C2 = []
        license_plates_C3 = []
        license_plates_J = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # names: [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgSrc.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    print(self._names[int(cls)])
                    if (self._names[int(cls)] == 'A' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_A.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'B' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_B.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'C' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_C.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'D' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_D.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'E' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_E.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'F' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_F.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'G' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_G.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'H' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_H.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))
                    if (self._names[int(cls)] == 'I' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_I.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))
                    if (self._names[int(cls)] == 'C1' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_C1.append(
                            ((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   3])),
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   3])))))
                    if (self._names[int(cls)] == 'C2' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_C2.append(
                            ((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   3])),
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   3])))))
                    if (self._names[int(cls)] == 'C3' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_C3.append(
                            ((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   3])),
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   3])))))
                    if (self._names[int(cls)] == 'J' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_J.append(
                            ((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   3])),
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   3])))))
        return sorted(license_plates_C3, key=lambda tup: [tup[0][1], tup[0][0]])

    def detect_license_plates(self, imgSrc, thresh=0.4):
        pred, img = self.predict_IMG(imgSrc)
        license_plates_A = []
        license_plates_B = []
        license_plates_C = []
        license_plates_D = []
        license_plates_E = []
        license_plates_F = []
        license_plates_G = []
        license_plates_H = []
        license_plates_I = []
        license_plates_C1 = []
        license_plates_C2 = []
        license_plates_C3 = []
        license_plates_J = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # names: [ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgSrc.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    print(self._names[int(cls)])
                    if (self._names[int(cls)] == 'A' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_A.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'B' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_B.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'C' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_C.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'D' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_D.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'E' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_E.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'F' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_F.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'G' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_G.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))

                    if (self._names[int(cls)] == 'H' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_H.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))
                    if (self._names[int(cls)] == 'I' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_I.append(((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       1])),
                                                                                                              (int(xyxy[
                                                                                                                       2]),
                                                                                                               int(xyxy[
                                                                                                                       3])),
                                                                                                              (int(xyxy[
                                                                                                                       0]),
                                                                                                               int(xyxy[
                                                                                                                       3])))))
                    if (self._names[int(cls)] == 'C1' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_C1.append(
                            ((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   3])),
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   3])))))
                    if (self._names[int(cls)] == 'C2' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_C2.append(
                            ((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   3])),
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   3])))))
                    if (self._names[int(cls)] == 'C3' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_C3.append(
                            ((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   3])),
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   3])))))
                    if (self._names[int(cls)] == 'J' and float(conf) > thresh):
                        cx = int(xyxy[0]) + (int(xyxy[2]) - int(xyxy[0])) / 2
                        cy = int(xyxy[1]) + (int(xyxy[3]) - int(xyxy[1])) / 2
                        license_plates_J.append(
                            ((int(xyxy[0]), int(xyxy[1]), cx, cy), self.croper.cropByCoor(imgSrc,
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   1])),
                                                                                          (int(xyxy[
                                                                                                   2]),
                                                                                           int(xyxy[
                                                                                                   3])),
                                                                                          (int(xyxy[
                                                                                                   0]),
                                                                                           int(xyxy[
                                                                                                   3])))))
        return sorted(license_plates_A, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_B, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_C, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_D, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_E, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_F, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_G, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_H, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_I, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_C1, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_C2, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_C3, key=lambda tup: [tup[0][1], tup[0][0]]), \
               sorted(license_plates_J, key=lambda tup: [tup[0][1], tup[0][0]])

    def detect_four_point(self, imgSrc, thresh=0.4):
        pred, img = self.predict_IMG(imgSrc)
        top_left = None
        top_right = None
        bottom_left = None
        bottom_right = None
        had_page = False
        had_page_tl = False
        had_page_tr = False
        had_page_bl = False
        had_page_br = False
        page_xmin = 0
        page_ymin = 0
        page_xmax = 0
        page_ymax = 0
        imgCrop_page = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], imgSrc.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    print(self._names[int(cls)])
                    if self._names[int(cls)] == 'page' and float(conf) > thresh:
                        had_page = True
                        page_xmin = int(xyxy[0])
                        page_ymin = int(xyxy[1])
                        page_xmax = int(xyxy[2])
                        page_ymax = int(xyxy[3])
                        imgCrop_page = imgSrc[page_ymin:page_ymax, page_xmin:page_xmax]
                    if (self._names[int(cls)] == 'point_1' and float(conf) > thresh):
                        had_page_tl = True
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        xcenter = xmin + (xmax - xmin) / 2
                        ycenter = ymin + (ymax - ymin) / 2
                        top_left = (xmin, ymin)

                    if (self._names[int(cls)] == 'point_2' and float(conf) > thresh):  # cic_tr
                        had_page_tr = True
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        xcenter = xmin + (xmax - xmin) / 2
                        ycenter = ymin + (ymax - ymin) / 2
                        top_right = (xmax, ymin)

                    if (self._names[int(cls)] == 'point_3' and float(conf) > thresh):  # cic_bl
                        had_page_bl = True
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        xcenter = xmin + (xmax - xmin) / 2
                        ycenter = ymin + (ymax - ymin) / 2
                        bottom_left = (xmax, ymax)

                    if (self._names[int(cls)] == 'point_4' and float(conf) > thresh):  # cic_br
                        had_page_br = True
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        xcenter = xmin + (xmax - xmin) / 2
                        ycenter = ymin + (ymax - ymin) / 2
                        bottom_right = (xmin, ymax)

            if (had_page_bl and had_page_br and had_page_tl and had_page_tr):
                # top_left = (page_xmin, page_ymin)
                # top_right = (page_xmax, page_ymin)
                return self.croper.cropByCoor(imgSrc, top_left, top_right, bottom_left, bottom_right)
            if had_page:
                return imgCrop_page

            else:
                return imgSrc