import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pathlib import Path
import pandas as pd
from matplotlib.patches import Polygon
from collections import Counter
import statistics
import pickle
from pprint import pprint
plt.gray()

def imgplot(img):
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.show()

def imgshow(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grayscale(img):
    return np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))


def resize_schema(img0,x,y,w,h,m,n):
    if len(img0.shape) == 3:
        img = img0.copy()[y:y+h,x:x+w,]
        result = np.zeros((m, n, 3), np.uint8)
    else:
        img = img0.copy()[y:y+h,x:x+w]
        result = np.zeros((m, n), np.uint8)
    
    r = img.shape[0]
    c = img.shape[1]
    rb = result.shape[0]
    cb = result.shape[1]

    hrb=round(rb/2)
    hcb=round(cb/2)
    hr=round(r/2)
    hc=round(c/2)

    up = hrb-hr
    bottom = hrb+hr
    left = hcb-hc
    right = hcb+hc
    
    
    if r % 2 == 1 and c % 2 == 0:
        up = hrb-hr+1
        
    elif r % 2 == 0 and c % 2 == 1:
        left = hcb-hc+1
            
    elif r % 2 == 1 and c % 2 == 1:
        up = hrb-hr+1
        left = hcb-hc+1
        
    if len(img0.shape) == 3:
        result[up:bottom,left:right,] = img
    else:
        result[up:bottom,left:right] = img
    
    return result, up, bottom, left, right



###data_path
data_path = Path().cwd().resolve().parent/'common'
probe_data_path = Path().cwd().resolve().parent / 'results'

###共通のシェーマの読み込み
with open(data_path/'schema_dict.pickle', 'rb') as dict:
    schema_dict = pickle.load(dict)

###Patientごとの情報の読み込み
patient_num = 4
df_probe_fixed = pd.read_pickle(probe_data_path/('patient' + str(patient_num) + '_probe_fixed.pickle'))
df_malignant_probe = pd.read_pickle(probe_data_path/('patient' + str(patient_num) + '_malignant_probe.pickle'))
df_nodule_probe = pd.read_pickle(probe_data_path/('patient' + str(patient_num) + '_nodule_probe.pickle'))


#Patientの外接矩形の座標
p_x, p_y = df_probe_fixed['(x,y)'].unique()[0]
p_w, p_h = df_probe_fixed['(w,h)'].unique()[0]
print(p_w,p_h)

#共通のシェーマの外接矩形の座標
schema_img = schema_dict['thy']
x,y = schema_dict['(x,y)']
w,h = schema_dict['(w,h)']
m,n = schema_dict['shape']
print(schema_dict)


#Step1 統一のシェーマから外接矩形の切り抜き
#schema_rect = schema_img[y:y+h,x:x+w]

#Step2 各患者の外接矩形に併せてリサイズ
#resize_img = cv2.resize(schema_rect, (p_w, p_h),interpolation=cv2.INTER_AREA)

#Step3 腫瘍をプロット
img_schema_nod = np.zeros((p_h, p_w, 3), np.uint8)
img_schema_mal = np.zeros((p_h, p_w, 3), np.uint8)

for i in range(len(df_nodule_probe)):
    #img_schema_part = img_schema.copy()
    nod_min = (df_nodule_probe['nod_min'][i][1] - p_x, df_nodule_probe['nod_min'][i][0] - p_y)
    nod_max = (df_nodule_probe['nod_max'][i][1] - p_x, df_nodule_probe['nod_max'][i][0] - p_y)
    print(nod_min,nod_max)
    cv2.line(img_schema_nod, nod_min, nod_max, (255,255,0))

imgplot(img_schema_nod)
print('==========')

for i in range(len(df_malignant_probe)):
    print(i)
    print(df_malignant_probe['image_name'][i])
    mal_min = (df_malignant_probe['nod_min'][i][1] - p_x, df_malignant_probe['nod_min'][i][0] - p_y)
    mal_max = (df_malignant_probe['nod_max'][i][1] - p_x, df_malignant_probe['nod_max'][i][0] - p_y)
    print(mal_min, mal_max)
    cv2.line(img_schema_mal, mal_min, mal_max, (0,255,255))
imgplot(img_schema_mal)


#Step4 元の外接矩形に変換
resize_nod= cv2.resize(img_schema_nod, (w, h), interpolation = cv2.INTER_CUBIC)
resize_mal= cv2.resize(img_schema_mal, (w, h), interpolation = cv2.INTER_CUBIC)

imgplot(resize_nod)
imgplot(resize_mal)

#Step5 切り抜いた部分と結合
result_nod, up, bottom, left, right  = resize_schema(resize_nod, 0, 0, w, h, m, n)
result_mal, up, bottom, left, right  = resize_schema(resize_mal,0, 0, w, h, m, n)

print(m,n)
print(result_nod.shape)

fin_nod = np.uint8(cv2.cvtColor(schema_img,cv2.COLOR_GRAY2BGR))
fin_mal = np.uint8(cv2.cvtColor(schema_img,cv2.COLOR_GRAY2BGR))

for i in range(m):
    for j in range(n):
        if (result_nod[i,j] > 0).any():
            fin_nod[i,j] = result_nod[i,j]

for i in range(m):
    for j in range(n):
        if (result_mal[i,j] > 0).any():
            fin_mal[i,j] = result_mal[i,j]

plt.imshow(fin_nod)
plt.savefig('p4mal.png')
plt.show()

plt.imshow(fin_mal)
plt.savefig('p4nod.png')
plt.show()



""" imgshow(result_nod)
imgshow(result_mal)
 """


""" 
plt.imshow(schema_img)
plt.savefig('all.png')
plt.show()
plt.imshow(schema_rect)
plt.savefig('a.png')
plt.show() 
plt.imshow(big)
plt.savefig('b.png')
plt.show()
cv2.namedWindow('schema', cv2.WINDOW_NORMAL)
cv2.imshow('schema',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
#誤差測定用
result[result < 135] = 0
result[result >= 135] = 255
im_diff = np.abs(result.astype(int) - schema_rect.astype(int))
plt.imshow(result)
plt.savefig('c.png')
plt.show()
plt.imshow(im_diff)
plt.savefig('d.png')
plt.show()
cv2.namedWindow('schema', cv2.WINDOW_NORMAL)
cv2.imshow('schema',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
    
