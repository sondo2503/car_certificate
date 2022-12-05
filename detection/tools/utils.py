import cv2

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def get_blurr_score(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return variance_of_laplacian(gray)

def get_blurr_score_from_cv2_file(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return variance_of_laplacian(gray)