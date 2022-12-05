import time

start_time = time.time()

from difflib import SequenceMatcher
from random import random
from difflib import SequenceMatcher
import csv
import urllib.parse
import random
import cv2
import os
import json
import xml.etree.ElementTree as ET
from OcrService import OcrService
from base.CustomException import *


# from imutils import paths

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def evaluate(img_path, pred, gt):
    result = {}
    result['img'] = img_path
    result['blurr_score'] = get_blurr_score(img_path)
    i = 0
    for k, v in gt.items():
        gt_value = v
        if isinstance(v, tuple):
            gt_value = v[0]
        print(pred[k])

        if k not in pred:

            result[k] = "0"
        if pred[k]=="":
            result[k] = ""
        else:
            pred_value = pred[k]
            if k == 'address':
                pred_value = ' '.join(list(pred[k].values()))
                # print(pred_value)
            if similar(str(pred_value).upper(), str(gt_value).upper()) > 0.9:
                result[k] = "1"
            else:
                result[k] = '0'
    return result


def read_json_gt(json_path):
    with open(json_path) as f:
        data = f.readlines()

    groundtruths = []
    for line in data:
        json_dict = json.loads(line)
        url_pic = json_dict['content']
        annotation = json_dict['annotation']
        gt = {}
        labels = {
            "E": "vin",
            "B": "registration_date",
            "A": "license_plate_number",
            "G": "vehicle_type",
            "C": "hsn",
            "D": "tsn",
            "F": "fuel_grade",
            "H": "emission_code",
            "I": "particulate_reduction_system"
        }
        try:
            for i in annotation:
                gt[labels[i['label'][0]]] = i['notes']

            if "/gen/" in url_pic:
                dataset_type = 'gen'
                encoded_pic_name = url_pic.split("/gen/")[-1]
            else:
                dataset_type = 'krug'
                encoded_pic_name = url_pic.rsplit("/")[-1]

            img_path = urllib.parse.unquote(encoded_pic_name)
            uuid = os.path.basename(img_path).rsplit(".", 1)[0]
            groundtruths.append((
                                os.path.join(r"C:\Users\Admin\Desktop\registration_text_xml_and_photos\registration_text_xml_and_photos\og_photos/",
                                             dataset_type, img_path), uuid, gt))
        except:
            pass
    return groundtruths
def rel(s):
    s = s.replace("[sp]","/")
    s = s.replace("[tp]",'\"')
    s = s.replace("[ta]","?")
    s = s.replace("[tb]",":")
    s = s.replace("[tc]","*")
    s = s.replace("[td]",'"')
    s = s.replace("[te]","<")
    s = s.replace("[tf]",">")
    s = s.replace("[tg]","|")
    return s

def read_xml_gt(xml_path):
    ROOT_IMG = r"C:\Users\Admin\Desktop\registration_text_xml_and_photos\registration_text_xml_and_photos\og_photos"
    labels = {
        "E": "vin",
        "B": "registration_date",
        "A Amtliches Kennzeichen": "license_plate_number",
        "5": "vehicle_type",
        "2.1": "hsn",
        "2.2": "tsn",
        "P.3": "fuel_grade",
        "14.1": "emission_code",
        "14": "particulate_reduction_system",
        "C.1.2 Vorname(n)": "first_name",
        "C.1.1 Name oder Firmenname": "name",
        "C.1.3 Anschrift": "address",
        "J": "vehicle_class"
    }
    groundtruths = []
    try:
        root = ET.parse(xml_path).getroot()

        gt = {}
        for obj_tag in root.findall('object'):
            name = obj_tag.find('name').text
            text = rel(obj_tag.find('text').text)
            print(text)
            ymin = obj_tag.find('bndbox').find('ymin').text

            if name == "2.2" and text:
                text = text.replace(" ", "")
            if name == "E" and text:
                text = text.replace("O", "0")
                text = text.replace("o", "0")
                text = text.replace("Q", "0")
                text = text.replace("L", "1")
                text = text.replace("l", "1")

            if name == "J" and text:
                text = text.replace("I", "1")
                text = text.replace("i", "1")

            if name in labels and text:
                if labels[name] in gt:
                    if text and ymin > gt[labels[name]][1]:
                        gt[labels[name]] = (gt[labels[name]][0] + " " + text, ymin)
                    else:
                        gt[labels[name]] = (text + " " + gt[labels[name]][0], gt[labels[name]][1])
                else:
                    gt[labels[name]] = (text, ymin)

        img_path = os.path.join(ROOT_IMG, root.find("filename").text)
        groundtruths.append((img_path, "", gt))
    except Exception as e:
        print(xml_path, e)
    return groundtruths


def process_img(ocr_service, img_path, uuid=""):
    img = cv2.imread(img_path)
    result = ocr_service._get__license_plates_from_trasf(img)
    return result.toDict()


def read_all_gt(gt_folder, type='json'):
    # gt_folder = "/home/ubuntu/projects/dataset/evaluation//gt"
    groundtruths = []
    for gt_file in os.listdir(gt_folder):
        # if "data_wintec_00250" not in gt_file:
        #     continue
        if type == 'json':
            groundtruths = groundtruths + read_json_gt(os.path.join(gt_folder, gt_file))
        if type == 'xml':
            groundtruths = groundtruths + read_xml_gt(os.path.join(gt_folder, gt_file))

    random.shuffle(groundtruths)
    return groundtruths


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def get_blurr_score(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return variance_of_laplacian(gray)


if __name__ == "__main__":
    ocr_service = OcrService()
    result = []

    # gt_folder = "/home/ubuntu/projects/dataset/evaluation//gt"
    # groundtruths = read_all_gt(gt_folder=gt_folder, type='json')

    gt_folder = r"C:\Users\Admin\Desktop\registration_text_xml_and_photos\registration_text_xml_and_photos\xml"
    groundtruths = read_all_gt(gt_folder=gt_folder, type='xml')

    i = 0
    for (img_path, _, gt) in groundtruths[0:]:
        # if "data_wintec_00000" not in img_path:
        #     continue
        # print(gt)
        i += 1
        # print("-----------------: " + str(i))
        try:
            predicted = process_img(ocr_service, img_path)
            print(predicted)
            result.append(evaluate(img_path, predicted, gt))

        except Exception as e:

            print(img_path, e)
            # raise(e)

    for i in result:
        keys = i.keys()
        if len(keys) == 15:
            break

    with open('eval26.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result)

print("--- %s seconds ---" % (time.time() - start_time))

# def get_field_gt(imgSrc, text_gt):
#     result = {}
#     for field, value in text_gt.items():
#         r_field = []
#         for gt in value:
#             [xmin, ymin, xmax, ymax] = gt['bbox']
#             top_left = (xmin, ymin)
#             top_right = (xmax, ymin)
#             bottom_right = (xmax, ymax)
#             bottom_left = (xmin, ymax)
#             cx = int((xmax-xmin)/2) + xmin
#             cy = int((ymax-ymin)/2) + ymin
#             r_field.append(((xmin, ymin, cx, cy), self.croper.cropByCoor(imgSrc, top_left, top_right, bottom_left, bottom_right))
#         result[field] = sorted(r_field, key=lambda tup: [tup[0][1], tup[0][0]])
#     return result