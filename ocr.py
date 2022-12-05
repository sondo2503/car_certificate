import csv
import difflib
import json

import cv2
from PIL import Image

from detection.detector import Detector
from recognition.vietocr.tool.config import Cfg
from recognition.vietocr.tool.predictor import Predictor

detect_pres = Detector(r"./weight/detection/final_detection_press.pt")
detect_pill = Detector(r"./weight/detection/final_detection_drug2.pt")
cfg_license_plates = Cfg.load_config_from_file(r'./recognition/vietocr/base.yml')

predict_pres = Predictor(cfg_license_plates)

def _cv2Pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def rel(s):
    s = s.replace('1)', '')
    s = s.replace('2)', '')
    s = s.replace('3)', '')
    s = s.replace('4)', '')
    s = s.replace('5)', '')
    s = s.replace('1.)', '')
    s = s.replace('2.)', '')
    s = s.replace('3.)', '')
    s = s.replace('5.)', '')
    s = s.replace('.', '')
    s = s.replace('-', '')
    s = s.replace('_', '')
    s = s.replace('%', '')
    s = s.replace(',', '')
    s = s.replace('+', '')
    s = s.replace(';', '')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace(':', '')
    s = s.replace('"', '')
    s = s.replace('0.5G', '500MG')
    s = s.replace('05G', '500MG')
    s = s.replace('0,5G', '500MG')
    s = s.replace("'", '')
    s = s.replace("DƯỞNG", 'DƯỠNG')
    s = s.replace("DƯỜNG", 'DƯỠNG')
    s = s.replace("DƯỚNG", 'DƯỠNG')
    s = s.replace("33DƯỠNG", 'DƯỠNG')
    s = s.replace("NÀO", 'NÃO')
    s = s.replace("11HOẠT", 'HOẠT')
    s = s.replace("22HOẠT", 'HOẠT')
    s = s.replace("22HOẠT", 'HOẠT')
    s = s.replace("2HOẠT", 'HOẠT')
    s = s.replace("3HOẠT", 'HOẠT')
    s = s.replace("3HOẠT", 'HOẠT')
    s = s.replace("DƯỚNG", 'DƯỠNG')
    s = s.replace("33HOẠT", 'HOẠT')
    s = s.replace("22GLU", 'GLU')
    s = s.replace("11GLU", 'GLU')
    s = s.replace("33GLU", 'GLU')
    s = s.replace("33SAV", 'SAV')
    s = s.replace("22SAV", 'SAV')
    s = s.replace("11CEF", 'CEF')
    s = s.replace("1CEF", 'CEF')
    s = s.replace("BỐ", 'BỔ')
    s = s.replace("MGT", 'MG')
    s = s.replace("5Đ", 'Đ')
    s = s.replace("11HAPEN", 'HAPEN')
    # s = s.replace("3X10500MGS", '3X10S')
    s = s.replace("HANGITOL", 'HANGITOR')
    s = s.replace("ENALAPRILT", 'ENALAPRIL')
    # s = s.replace("30500MGS", '30S')
    # s = s.replace("60500MGS", '60S')
    s = s.replace("22PAR", 'PAR')
    s = s.replace("22VINA", 'VINA')
    s = s.replace("11NOVO", 'NOVO')
    s = s.replace("22DIAM", 'DIAM')
    s = s.replace("1ĐINH", 'ĐINH')
    s = s.replace("2ĐINH", 'ĐINH')
    s = s.replace("3ĐINH", 'ĐINH')
    s = s.replace("33PAR", 'PAR')
    s = s.replace("33VIP", 'VIP')
    s = s.replace("MG11", 'MG1')
    s = s.replace("33COSY", 'COSY')
    s = s.replace("33STA", 'STA')
    s = s.replace("22MYP", 'MYP')
    s = s.replace('5.)', '')
    s = s.replace('5)', '')
    s = s.replace('.jpg', '')
    return s

def predict_press(lst):
    content = []
    for i, a in enumerate(lst):
        im_pil = _cv2Pil(a)
        s = predict_pres.predict(im_pil, return_prob=False)
        s = rel(s).strip()
        content.append(rel(s).strip())

    return content

def recognition_press(img):
    Text = detect_pres.detection_press(img)
    text_new = predict_press(Text)
    return text_new


def detection_durg(img):
    durg_predict = detect_pill.detect_drug(img)
    return durg_predict

import os.path as osp
map_press = r'pill_pres_map.json'
path_pres_ocr = r'D:\DATA_COMPE\RELEASE_private_test\prescription\image'
path_pill_detection = r'D:\DATA_COMPE\RELEASE_private_test\pill\image'
with open(osp.join(map_press)) as f:
    data = json.load(f)
    for i in data:
        image_pres_ocr = osp.join(path_pres_ocr, i + str(".png"))

        imagge_prdict = cv2.imread(image_pres_ocr)
        text_new = recognition_press(imagge_prdict)
        list_box = []
        with open(r'id_drug.txt', encoding='utf-8') as file:
            for line in file:
                list_box.append(line.rstrip())
        list_pill = []
        for x, a in enumerate(list_box):

            for a_text in range(len(text_new)):
                text1 = text_new[a_text].upper()
                text_predict = ''.join(text1.split())
                text_predict_1 = rel(text_predict)
                text_2 = list_box[x].split('\t')[0].upper()
                text_predict_c = ''.join(text_2.split())
                text_predict_2 = rel(text_predict_c)

                if int(difflib.SequenceMatcher(None, text_predict_1, text_predict_2).ratio() * 100) > 90:
                    list_pill.append(list_box[x].split('\t')[1])
                    print(text_predict_1, text_predict_2)
                    break
        for image_durg in data[i]:
            image_path_detect = osp.join(path_pill_detection, image_durg )
            print(image_path_detect)
            imagge_prdict_durg = cv2.imread(image_path_detect)
            Text_new = detection_durg(imagge_prdict_durg)

            for i_tem in Text_new:

                a, b, c, d, e, f = i_tem
                item_new = list(i_tem)

                if str(int(float(a))) not in list_pill:
                    item_new[0] = 107
                item_new.insert(0, image_durg.replace('.json', ".jpg"))
                print(item_new)
                txt_path2 = r'results.csv'
                with open(txt_path2, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(item_new)
