'''
出力がjpgなので，入力した画像を完全に復元しているわけではないことに注意！！！
'''

from pathlib import Path
from pprint import pprint
import re
import cv2

patient_number = input()
patient_path = 'Patient ' + patient_number
data_path = Path().cwd() / patient_path

image_list = data_path.glob('*Image*.jpg') 

for image_path in image_list:
    #print(image_path)
    img = cv2.imread(str(image_path))[40:-40,20:-20]
    cv2.imwrite(str(image_path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])