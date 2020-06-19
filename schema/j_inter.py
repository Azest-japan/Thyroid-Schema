#data
intersection = [['v', (185, 190, 247, 300)],['v', (185, 190, 124, 194)],
['h', (49, 98, 289, 294)],['v', (185, 190, 318, 477)],
['h', (81, 179, 431, 436)],['v', (153, 158, 283, 389)],
['v', (153, 158, 265, 283)],['v', (153, 158, 177, 230)],
['v', (299, 304, 318, 459)],['v', (185, 190, 283, 442)],
['v', (185, 190, 177, 247)],['v', (478, 483, 247, 300)],
['v', (478, 483, 424, 512)],['h', (423, 456, 395, 400)],
['v', (446, 451, 353, 406)]]

def calc_inter(intersection):
    inter_list=[]

    h_list = [k[1] for k in intersection if k[0] == 'h']
    v_list = [k[1] for k in intersection if k[0] == 'v']

    #print(h_list)
    #print(v_list)

    for i, (h_x1, h_x2, h_y1, h_y2) in enumerate(h_list):
        h_xran = list(range(h_x1, h_x2+1))
        h_yran = list(range(h_y1, h_y2+1))
    
        for j, (v_x1, v_x2, v_y1, v_y2) in enumerate(v_list):
            v_xran = list(range(v_x1, v_x2+1))
            v_yran = list(range(v_y1, v_y2+1))

            target_xran = set(h_xran) & set(v_xran)
            target_yran = set(h_yran) & set(v_yran)

            #print(i,j)
            #print(target_xran)
            #print(target_yran)            
            if len(target_xran) > 0 and len(target_yran) > 0:
                #print('True')
                tar_x1 = min(target_xran)
                tar_x2 = max(target_xran)
                tar_y1 = min(target_yran)
                tar_y2 = max(target_yran)
            
                inter_list.append((int((tar_x1 + tar_x2)/2), int((tar_y1 + tar_y2)/2)))
    
    return list(set(inter_list))


inter_list = calc_inter(intersection)
print(inter_list)


"""
#以下plot用
import pandas as pd
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
def imgplot(img):
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.show()



schema_img = cv2.imread(r'C:\Users\TatsuyaKawakami\Downloads\Thyroid-Schema-master\schema\test\Ito\schema2.jpg')
imgplot(schema_img)

color_list = [(255,255,0),(255,0,255)]
for i in intersection:
    x1,x2,y1,y2 = i[1]
    cv2.rectangle(schema_img, (x1,y1), (x2,y2), color_list[0],thickness=-1)

imgplot(schema_img)

for i in inter_list:
    x,y = i
    cv2.line(schema_img, (x,y), (x,y), color_list[1])

imgplot(schema_img)

"""