import base64
import time

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import boto3
import shutil
import glob

from detection.detector import *
from base.ErrorCode import ErrorCode
from schemas.license_plates_reponse import CertificateContent_license_plates
from base.CustomException import *
from tools.utils import *
from recognition.vietocr.tool.config import Cfg
from recognition.vietocr.tool.predictor import Predictor

license_plates_detector = Detector(r"./weight/detection/best.pt")
license_plates_corner_detector = Detector(r"weight/corner_detection/best_corner.pt")
cfg_license_plates = Cfg.load_config_from_file(r'./recognition/vietocr/base.yml')

predictor_license_plates = Predictor(cfg_license_plates)

class OcrService:
    def _compute_hist(self, img):
        hist = np.zeros((256,), np.uint8)
        h, w = img.shape[:2]
        for i in range(h):
            for j in range(w):
                hist[img[i][j]] += 1
        return hist

    def _equal_hist(self, hist):
        cumulator = np.zeros_like(hist, np.float64)
        for i in range(len(cumulator)):
            cumulator[i] = hist[:i].sum()
        # print(cumulator)
        new_hist = (cumulator - cumulator.min()) / (cumulator.max() - cumulator.min()) * 255
        new_hist = np.uint8(new_hist)
        return new_hist

    def _balance(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = self._compute_hist(img).ravel()
        new_hist = self._equal_hist(hist)
        h, w = img.shape[:2]
        for i in range(h):
            for j in range(w):
                img[i, j] = new_hist[img[i, j]]
        return img

    def _balanceHistoram(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    def _read_image_b64(self, base64_string):
        base64_string = base64_string.split(',')[1]
        jpg_original = base64.b64decode(base64_string)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=1)
        return img

    # $%&''()*+,-./:;<=>?@[\]^_`{|}~
    def _replace_to_bar(self, s=''):
        s = s.replace('/', '-')
        s = s.replace('?', '-')
        s = s.replace(' ', '-')
        s = s.replace('.', '-')
        s = s.replace('~', '-')
        s = s.replace('@', '-')
        s = s.replace('\\', '-')
        s = s.replace('"', '-')
        s = s.replace('%', '-')
        s = s.replace('#', '-')
        s = s.replace("c'", 'Ć')
        return s

    def _replace_J(self, s):
        s = s.replace('I', '1')

        return s
    def _replace_vin(self, s):
        s = s.replace('o', '0')
        s = s.replace('O', '0')
        s = s.replace('O', '0')
        s = s.replace('L', '1')
        s = s.replace('l', '1')
        s = s.replace('q', '0')
        s = s.replace('Q', '0')
        return s
    def _replace_H(self, s):
        s = s.replace('O', '0')
        s = s.replace('o', '0')

        return s

    def _cv2Pil(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil

    def _calConfAndStringTransfomer_license_plates(self, lst):
        if not lst:
            return {}
        conf = 0.0
        count = 0
        content = ''
        for i, a in enumerate(lst):
            out = a[1]
            im_pil = self._cv2Pil(out)
            s, c = predictor_license_plates.predict(im_pil, return_prob=True)
            content += s + ' '

            count += 1
            conf += float(c)
        return {"content": content.strip(), "conf": round((conf / count), 4), }
    def _calConfAndStringTransfomer_license_plates_H(self, lst):
        if not lst:
            return {}
        conf = 0.0
        count = 0
        content = ''
        for i, a in enumerate(lst):
            out = a[1]
            im_pil = self._cv2Pil(out)
            s, c = predictor_license_plates.predict(im_pil, return_prob=True)
            content += self._replace_H(s) + ' '

            count += 1
            conf += float(c)
        return {"content": content.strip(), "conf": round((conf / count), 4), }

    def _calConfAndStringTransfomer_license_plates_vin(self, lst):
        if not lst:
            return {}
        conf = 0.0
        count = 0
        content = ''
        for i, a in enumerate(lst):
            out = a[1]
            im_pil = self._cv2Pil(out)
            s, c = predictor_license_plates.predict(im_pil, return_prob=True)
            content += self._replace_vin(s) + ' '

            count += 1
            conf += float(c)
        return {"content": content.strip(), "conf": round((conf / count), 4), }

    def _calConfAndStringTransfomer_J(self, lst):
        conf = 0.0
        count = 0
        content = ''
        for i, a in enumerate(lst):
            out = a[1]
            im_pil = self._cv2Pil(out)
            s, c = predictor_license_plates.predict(im_pil, return_prob=True)
            content += self._replace_J(s) + ' '

            count += 1
            conf += float(c)
        return {"content": content.strip(), "conf": round((conf / count), 4), }

    def _calConfAndStringTransfomer_license_plates_page(self, lst):
        conf = 0.0
        count = 0
        content = ''

        # im_pil = self._cv2Pil(lst)
        s, c = predictor_license_plates.predict(lst, return_prob=True)
        content += s + ' '

        count += 1
        conf += float(c)
        return {"content": content.strip(), "conf": round((conf / count), 4), }

    def _get__license_plates_from_trasf(self, img):
        try:
            bf = time.time()
            # print(bf)
            image_new = license_plates_corner_detector.detect_four_point(img)
            image_h, image_w, c = image_new.shape
            print(image_w, image_h)
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

            if (image_w > image_h):
                license_plates_A, license_plates_B, license_plates_C, license_plates_D, license_plates_E, \
                license_plates_F, license_plates_G, license_plates_H, license_plates_I, license_plates_C1, license_plates_C2, license_plates_C3, license_plates_J = license_plates_detector.detect_license_plates(
                    image_new)

                if len(license_plates_J):
                    registration_date = self._calConfAndStringTransfomer_license_plates(license_plates_E)

                    if license_plates_E == [] or registration_date['conf'] < 0.69:
                        image_new4 = ndimage.rotate(image_new, 180)
                        license_plates_A, license_plates_B, license_plates_C, license_plates_D, license_plates_E, \
                        license_plates_F, license_plates_G, license_plates_H, license_plates_I, license_plates_C1, license_plates_C2, license_plates_C3, license_plates_J = license_plates_detector.detect_license_plates(
                            image_new4)

                else:
                    license_plates_A, license_plates_B, license_plates_C, license_plates_D, license_plates_E, \
                    license_plates_F, license_plates_G, license_plates_H, license_plates_I, license_plates_C1, license_plates_C2, license_plates_C3, license_plates_J = license_plates_detector.detect_license_plates(
                        img)

            if (image_w < image_h):

                license_plates_A, license_plates_B, license_plates_C, license_plates_D, license_plates_E, \
                license_plates_F, license_plates_G, license_plates_H, license_plates_I, license_plates_C1, license_plates_C2,license_plates_C3,  license_plates_J = license_plates_detector.detect_license_plates(
                    img)

                registration_date = self._calConfAndStringTransfomer_license_plates(license_plates_E)

                # print(registration_date['conf'])
                try:
                    if license_plates_E == [] or registration_date['conf'] < 0.69:
                        image_new_2 = ndimage.rotate(image_new, 90)
                        license_plates_A, license_plates_B, license_plates_C, license_plates_D, license_plates_E, \
                        license_plates_F, license_plates_G, license_plates_H, license_plates_I, license_plates_C1, license_plates_C2, license_plates_C3, license_plates_J = license_plates_detector.detect_license_plates(
                            image_new_2)

                        registration_date = self._calConfAndStringTransfomer_license_plates(license_plates_E)

                        if license_plates_E == [] or registration_date['conf'] < 0.69:
                            image_new3 = ndimage.rotate(image_new_2, 180)
                            license_plates_A, license_plates_B, license_plates_C, license_plates_D, license_plates_E, \
                            license_plates_F, license_plates_G, license_plates_H, license_plates_I, license_plates_C1, license_plates_C2,license_plates_C3,  license_plates_J = license_plates_detector.detect_license_plates(
                                image_new3)

                    else:
                        license_plates_A, license_plates_B, license_plates_C, license_plates_D, license_plates_E, \
                        license_plates_F, license_plates_G, license_plates_H, license_plates_I, license_plates_C1, license_plates_C2,license_plates_C3,  license_plates_J = license_plates_detector.detect_license_plates(
                            img)

                except:
                    pass

                # else:
                #     license_plates_A, license_plates_B, license_plates_C, license_plates_D, license_plates_E, \
                #     license_plates_F, license_plates_G, license_plates_H, license_plates_I, license_plates_C1, license_plates_C2, license_plates_C3, license_plates_J = license_plates_detector.detect_license_plates(
                #         img)
            license_plate_number = self._calConfAndStringTransfomer_license_plates(license_plates_A)
            registration_date = self._calConfAndStringTransfomer_license_plates(license_plates_E)
            hsn = self._calConfAndStringTransfomer_license_plates(license_plates_C)
            tsn = self._calConfAndStringTransfomer_license_plates(license_plates_D)
            vin = self._calConfAndStringTransfomer_license_plates_vin(license_plates_B)

            fuel_grade = self._calConfAndStringTransfomer_license_plates(license_plates_F)
            vehicle_type = self._calConfAndStringTransfomer_license_plates(license_plates_G)
            emission_code = self._calConfAndStringTransfomer_license_plates_H(license_plates_H)
            particulate_reduction_system = self._calConfAndStringTransfomer_license_plates(license_plates_I)
            license_plates_c1 = self._calConfAndStringTransfomer_license_plates(license_plates_C1)
            license_plates_c2 = self._calConfAndStringTransfomer_license_plates(license_plates_C2)
            license_plates_c3 = self._calConfAndStringTransfomer_license_plates(license_plates_C3)
            license_plates_j = self._calConfAndStringTransfomer_J(license_plates_J)

            clp = CertificateContent_license_plates(license_plate_number, registration_date, hsn, tsn, vin, fuel_grade,
                                                    vehicle_type, emission_code, particulate_reduction_system,
                                                    license_plates_c1, license_plates_c2, license_plates_c3,
                                                    license_plates_j)
            if not clp.isEmpty():
                return clp
            else:
                raise Exception("Not detect image")
        except Exception as e:
            raise

    def get_fzs_feedback_from_cus(self, item):
        return item.data.dict(exclude_none=True)

    def get_fzs_content(self, item, blurr_threshold=100):
        img_path = self._download_img_from_S3(item)
        img = self._read_img_from_path(img_path)
        blurr_score = get_blurr_score_from_cv2_file(img)
        if blurr_score < blurr_threshold:
            raise BlurrImgException("Image is too blurr. Please retake !")
        result = self._get__license_plates_from_trasf(img)
        return result.toDict()

    def _read_img_from_path(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                Image.open(img_path).convert("RGB").save(img_path)
                img = cv2.imread(img_path)
            return img
        except Exception:
            raise Exception("Unable to read the image")

    def _download_img_from_S3(self, item):
        try:
            uuid = item.uuid
            bucket = item.bucket
            img_path = self._aws2image(uuid, bucket)
            return img_path
        except Exception:
            raise Exception("Unable to download the image")

    def _aws2image(self, uuid, bucket):
        save_folder = os.path.join('data/images', uuid)
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder, exist_ok=True)

        s3 = boto3.resource('s3')
        my_bucket = s3.Bucket(bucket)
        prefix = "vehicle-registration" + '/' + uuid + '/'
        for s3_object in my_bucket.objects.filter(Prefix=prefix):
            path, filename = os.path.split(s3_object.key)
            print("%%%%%%%%%%%%%")
            print(filename)
            if "medium" in filename or 'original' in filename:
                my_bucket.download_file(
                    s3_object.key, os.path.join(save_folder, filename))
                break

        image_paths = glob.glob(os.path.join('data/images', uuid) + '/*')

        if len(image_paths) > 0:
            return image_paths[0]
        return None

    def _get__license_plates_page_from_trasf(self, img):
        bf = time.time()
        # print(bf)
        page_name = license_plates_corner_detector.detect_page(img)
        list_blurr_score = []
        for name, page in page_name:
            blurr_score_page = get_blurr_score_from_cv2_file(page)
            list_blurr_score.append(blurr_score_page)
        return list_blurr_score

    def get_license_plates_content(self, image, blurr_threshold=100):
        if image.file is None:
            return ErrorCode("Please upload a valid image")
        imgStr = image.file.read()
        if imgStr is None or imgStr == b'':
            return ErrorCode("Cannot read the image: " + image.filename)
        npimg = np.fromstring(imgStr, np.uint8)
        img = cv2.imdecode(npimg, flags=1)

        blurr_score_all = self._get__license_plates_page_from_trasf(img)
        # print(blurr_score)
        print("----------")
        for blurr_score in blurr_score_all:
            if blurr_score < blurr_threshold:
                raise BlurrImgException("Image is too blurr. Please retake !")
        try:
            return self._get__license_plates_from_trasf(img)
        except:
            return ErrorCode("Not detect image")

    def get_license_plates_page_content(self, image):
        if image.file is None:
            return ErrorCode("Please upload a valid image")
        imgStr = image.file.read()
        if imgStr is None or imgStr == b'':
            return ErrorCode("Cannot read the image: " + image.filename)
        npimg = np.fromstring(imgStr, np.uint8)
        img = cv2.imdecode(npimg, flags=1)
        return self._get__license_plates_page_from_trasf(img)


