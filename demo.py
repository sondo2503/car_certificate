from evaluation import process_img
from OcrService import OcrService

ocr_service = OcrService()
img_path = "data_wintec_01648.jpg"

result = process_img(ocr_service, img_path)
print(result)