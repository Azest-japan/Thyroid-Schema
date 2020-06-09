import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import gc
import os
from tqdm.notebook import tqdm
from pathlib import Path
import pandas as pd
from matplotlib.patches import Polygon
from collections import Counter
import statistics
import pickle

from math import floor
from decimal import Decimal, ROUND_HALF_UP
import pprint

from PatientClass import cut
plt.gray()


### ユーティリティ
#画像の表示

def imgshow(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imgplot(img):
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.show()
    
def grayscale(img):
    return np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))


# 孤立しているピクセルを取り除く
def clean_image(img):
    m,n = img.shape
    for i in range(1,m-1):
            for j in range(1,n-1):
                if (img[i,j] > 0 and img[i,j-1] == 0 and img[i,j+1] == 0 and img[i-1,j-1] == 0
                and img[i-1,j+1] == 0 and img[i+1,j-1] == 0 and img[i+1,j+1] == 0):
                    img[i,j] = 0
    return img


# 画像の読み込み及び辞書の作成
def make_image_dic(image_namelist):
    image_dic = {}
    for i in image_namelist:
        img0 = cv2.imread(str(i))
        print(i.stem,':',img0.shape)
        image_dic[i.stem] = img0
    return image_dic

def image_to_name(img,annotated_image_dic,return_name):
    image_name = [k for k, v in annotated_image_dic.items() if (v == img).all()]
    if return_name == True:
        return image_name[0]
    else:
        print('image_name:',image_name[0])

def annotated_to_image(img,annotated_image_dic,image_dic):
    annotated_image_name = image_to_name(img,annotated_image_dic,True)
    image_name = annotated_image_name.replace('annotated','')
    img0 = image_dic[image_name]
    return img0



#整数かどうかの判定
def is_integer_num(n):
    if isinstance(n, int):
        return 1
    if isinstance(n, float):
        if n.is_integer() == True:
            return 1
        else: return 0
    return False


# ## スキーマの作成及び外接矩形の算出




#エコー画像からスキーマ部分の特定
def thyroid_detect(img):
    img_gray = grayscale(img)
    #imgplot(img_gray)
    
#edge
    c = cv2.Canny(img_gray,650,800)
    #imgplot(c)
    
#フィルタ
    kernel = np.ones((50,50),np.float32)/500
    dst = cv2.filter2D(c,-1,kernel,borderType = cv2.BORDER_WRAP)
    #imgplot(dst)

#中心を特定
    ct = np.int32(np.mean(np.where(dst==np.max(dst)),axis=-1))
    print('ct:',ct)
    
    return dst,ct

#エコー画像からスキーマ部分の切り出し
def thyroid_trim(img,dst,ct):
    t,b,l,r = 0,0,0,0
    s = 0
    while(s<4):
        s = 0
        if  dst[ct[0]-t,ct[1]] >= 85:
            t += 1
        else:
            s += 1
        if  dst.shape[0]>ct[0]+b and dst[ct[0]+b,ct[1]] >= 88:
            b += 1
        else:
            s += 1
        if  dst[ct[0],ct[1]-l] >= 88:
            l += 1
        else:
            s += 1
        if  dst.shape[1]>ct[1]+r and dst[ct[0],ct[1]+r] >= 88:
            r += 1
        else:
            s += 1

    #print('t:',t,',b:',b,',l:',l,',r',r)
    print('[{}:{},{}:{}]'.format(ct[0]-t,ct[0]+b,ct[1]-l,ct[1]+r))
    sc = img[ct[0]-t:ct[0]+b,ct[1]-l:ct[1]+r].copy()
    #print('sc:')
    #imgplot(sc)
    
    #sc0 = grayscale(sc)
    #sc0[sc0<100] = 0
    #sc[sc0==0] = 0
    #print('sc0')
    #imgplot(sc0)
    print('sc')
    imgplot(sc)
    
    return sc,[t,b,l,r]




#スキーマ→プローブの抽出
def probe_detect(sc):
    m,n,d = sc.shape
    sh = np.uint8(np.zeros((sc.shape[0],sc.shape[1])))

    for i in range(m):
        for j in range(n):
            b,g,r = sc[i,j]
            if (150 <= r and 80 <= g and g <= 190 and b <=110 and g > b) or (r<220 and r<b-15 and r<g-10):
                sh[i,j] = 255
    print('image sh:')
    imgplot(sh)
    
    if np.max(sh.sum(axis=0)) > np.max(sh.sum(axis=1)):
        direction = 'vertical'
    else: direction = 'horizonal'
    print('direction',direction)
    return sh, direction





#スキーマの甲状腺の中心の列方向の座標，プローブの座標計算
def thyroid_measure(sc,sh,direction):
    dmin = 100
    cmax = int(sc.shape[1]/2)-5

    sc0 = grayscale(sc)
    sc0[sh>0] = 0
    
    #ノイズを除去
    #m,n = sc0.shape 
    #sc0 = np.where(sc0<10,0,sc0)
    #for i in range(1,m-1):
    #        for j in range(1,n-1):
    #            if (sc0[i,j] > 0 and sc0[i,j-1] == 0 and sc0[i,j+1] == 0 and sc0[i-1,j-1] == 0
    #            and sc0[i-1,j+1] == 0 and sc0[i+1,j-1] == 0 and sc0[i+1,j+1] == 0):
    #                sc0[i,j] = 0
    
    
    sc_copy = sc.copy()

    print('sc0')
    imgplot(sc0)
    print(sc0.shape)

    for i in range(int(sc.shape[1]/2)-5,int(sc.shape[1]/2)+6):
        col = len(sc0[:,i])

        p = 0
        n = 0
        j = 10
        while j<col:
            #print(i,' ',j,' ',sc0[j,i],' ',sc0[j-2,i],' ',p,' ',n,' ',cmax,' ',dmin)
        
            if (sc0[j,i] > 120) and (sc0[j-2,i] < 36) :
                n += 1
                if n == 2:
                    p = j
                elif n==3:
                    if j-p < dmin:
                        dmin = j-p
                        cmax = i
                    #print('break ',i,' ',j,' ',sc0[j,i],' ',sc0[j-2,i],' ',p,' ',n,' ',cmax,' ',dmin,' ',j-p)
                    break
                j += 3
            j += 1
    
    sc_copy[:,cmax] = [200,100,50]
    
    sh_1 = sh.copy()
    sh_1_m, sh_1_n = sh_1.shape    
    sh_1 = clean_image(sh_1)
    print('sh_1')
    imgplot(sh_1)

    
    if direction =='vertical':
        cd = np.argmax(sh.sum(axis=0))
        print('cd:',cd)
        th_min, th_max = probe_measure(sh, cd, direction)
        print('th_min:{},th_max:{}'.format(th_min,th_max))
        for i in range(sh_1_m):
            for j in range(sh_1_n):
                if sh_1[i,j] > 0 and (i < th_min or i > th_max):
                    sh_1[i,j] = 0
        
        col_list = np.count_nonzero(sh_1 > 0,axis=0)
        col_max = max(col_list)

        col_num_list = []
        for i,col_len in enumerate(col_list):
            if i == 0:
                if col_len > 0.6 * col_max and col_list[i+1] > 0.6 * col_max:
                    col_num_list.append(i)
            elif i == len(col_list) - 1:
                if col_len > 0.6 * col_max and col_list[i-1] > 0.6 * col_max:
                    col_num_list.append(i)
            else:
                if col_len > 0.6 * col_max and (col_list[i-1]> 0.6 * col_max or col_list[i+1] > 0.6 * col_max):
                    col_num_list.append(i)
                    
        print(col_num_list)
        cd = int(statistics.median(col_num_list))
        sc_copy[:,cd] = [50,200,50]
        
    elif direction == 'horizonal':
        cd = np.argmax(sh.sum(axis=1))
        th_min, th_max = probe_measure(sh, cd, direction)
        
        print('th_min:{},th_max:{}'.format(th_min,th_max))
        
        for i in range(sh_1_m):
            for j in range(sh_1_n):
                if sh_1[i,j] > 0 and (j < th_min or j > th_max):
                    sh_1[i,j] = 0

        row_list = np.count_nonzero(sh_1 > 0,axis=1)
        row_max = max(row_list)
        
        print('row_list:',row_list)
        print('row_max:',row_max)
        row_num_list = []
        for i,row_len in enumerate(row_list):
            if i == 0:
                if row_len > 0.6 * row_max and row_list[i+1] > 0.6 * row_max:
                    row_num_list.append(i)
                    
            elif i == len(row_list) - 1:
                if row_len > 0.6 * row_max and row_list[i-1] > 0.6 * row_max:
                    row_num_list.append(i)
            else:
                if row_len > 0.6 * row_max and (row_list[i-1]> 0.6 * row_max or row_list[i+1] > 0.6 * row_max):
                    row_num_list.append(i)

        cd = int(statistics.median(row_num_list))     
        sc_copy[cd,:] = [50,200,50]
    
    print('sc_copy')
    imgplot(sc_copy)
    return cmax, cd

#プローブの残りの座標計算
def probe_measure(sh,cd,derection):
    sh = clean_image(sh)
    if derection== 'vertical':
        sh_hot = sh[:,cd]
    elif derection=='horizonal':
        sh_hot = sh[cd,:]
#thyroidのcol
    th_max = 0
    th_min = 0
    for (i, x) in enumerate(sh_hot):
        if x>0 and th_min==0:
            th_min = i
        elif x==0 and th_min>0 and th_max==0:
            th_max = i-1
    return th_min, th_max





#スキーマからプローブを削除，ノイズを除去
def clean_schema(sc,sh,direction):
#探索用甲状腺画像を作成
    sc1 = grayscale(sc)
    sc1[sh>0] = 0
    #imgplot(sc1)
    m,n = sc1.shape
    
    
    #ノイズを除去
    m,n = sc1.shape 
    sc1 = np.where(sc1<10,0,sc1)
    for i in range(1,m-1):
            for j in range(1,n-1):
                if (sc1[i,j] > 0 and sc1[i,j-1] == 0 and sc1[i,j+1] == 0 and sc1[i-1,j-1] == 0
                and sc1[i-1,j+1] == 0 and sc1[i+1,j-1] == 0 and sc1[i+1,j+1] == 0):
                    sc1[i,j] = 0
    
    # 縦方向にエコーを見ていた場合の線の残りを消去
    if direction =='vertical':
        for i in range(1,m-1):
            for j in range(1,n-1):
                if (sc1[i,j] > 20 and sc1[i,j-1] < 20 and sc1[i,j+1] < 20 and sc1[i-1,j-1] < 20
                and sc1[i-1,j+1] < 20 and sc1[i+1,j-1] < 20 and sc1[i+1,j+1] < 20):
                    sc1[i,j] = 0
                
    #画像の端にあるノイズを削除
    #縦
    for i in range(m):
        if sc1[i,0]>20 and sc1[i,1]<20:
            sc1[i,0] = 0
        elif sc1[i,n-1]>20 and sc1[i,n-2]<20:
            sc1[i,n-1] = 0

    #横
    for i in range(n):
        if sc1[0,i]>20 and sc1[1,i]<20:
            sc1[0,i] = 0
        elif sc1[m-1,i]>20 and sc1[m-2,i]<20:
            sc1[m-1,i] = 0
    print('sc1:')
    imgplot(sc1)
    return sc1

'''
elif direction == 'horizonal':
    for i in range(1,m-1):
        for j in range(n):
            if sc1[i,j] > 20 and sc1[i-1,j] < 10 and sc1[i-1,j] < 10:
                sc1[i,j] = 0    
''' 





#スキーマから不要な部分を削除
def remove_neck(sc1,cmax):
    #上側の消去
    m_sc1,n_sc1 = sc1.shape
    ur_temp = sc1[:,cmax]

    n=0
    neck=0
    cen_row_max = 0
    cen_row_min = 0

    for (i, x) in enumerate(ur_temp):
        if x>=30 and n==0:
            n+=1
            neck = i
        elif x>=30 and n==1:
            neck = i
        elif x<30 and n==1:
            n+=1
        elif x>=100 and n==2 and cen_row_min==0:
            cen_row_min = i
            n+=1
        elif x<100 and n==3:
            n+=1
        elif x>=100 and n==4:
            n+=1
            cen_row_max = i
        elif x>=100 and n==5:
            cen_row_max = i

    sc2 = sc1.copy()
    sc2[:neck+1,:]=0
    print('head')
    imgplot(sc2)

    #左右の探索
    left = 0
    right = n_sc1

    ur_copy = sc2.copy()
    
    ur_copy[cen_row_min:,] = 0
    
    ur_temp = np.count_nonzero(ur_copy > 20, axis = 0)
    print(ur_temp)
    n = 0
    for (i, x) in enumerate(ur_temp):
        if x > 0 and n == 0:
            n = 1
            left = i
        elif x > 0 and n == 1:
            left = i
        elif x == 0 and n == 1:
            n += 1
            break


    n = 0
    for (i, x) in enumerate(reversed(ur_temp)):
        if x > 0 and n == 0:
            n = 1
            right = n_sc1 - 1-i
        elif x > 0 and n == 1:
            right = n_sc1 - 1-i
        elif x == 0 and n == 1:
            n += 1
            break

    sc2[:,:left+1]=0
    sc2[:,right:]=0

    imgplot(sc2)
    
    #下の探索
    ub_copy = sc2.copy()
    ub_temp = np.count_nonzero(ub_copy > 20, axis = 1)
    n = 0
    for (i, x) in enumerate(reversed(ub_temp)):
        if x > 0 and n == 0:
            n = 1
            bottom = m_sc1 - 1-i
        elif x > 0 and n == 1:
            bottom = m_sc1 - 1-i
        elif x == 0 and n == 1:
            n += 1
            break
    
    #print('bottom:',bottom)
    if cen_row_max < bottom or cen_row_min < bottom:
        sc2[bottom:,:]=0

    #ノイズを除去
    sc2 = np.where(sc2<10,0,sc2)
    for i in range(1,m_sc1-1):
            for j in range(1,n_sc1-1):
                if (sc2[i,j] > 0 and sc2[i,j-1] == 0 and sc2[i,j+1] == 0 and sc2[i-1,j-1] == 0
                and sc2[i-1,j+1] == 0 and sc2[i+1,j-1] == 0 and sc2[i+1,j+1] == 0):
                    sc2[i,j] = 0

    print('sc2')
    imgplot(sc2)
    
    return cen_row_min,cen_row_max,sc2





#輪郭描写
def draw_contours(ax, img, contours):
    ax.imshow(img)  # 画像を表示する。
    ax.set_axis_off()

    for i, cnt in enumerate(contours):
        # 形状を変更する。(NumPoints, 1, 2) -> (NumPoints, 2)
        cnt = cnt.squeeze(axis=1)
        # 輪郭の点同士を結ぶ線を描画する。
        ax.add_patch(Polygon(cnt, color="b", fill=None, lw=2))
        # 輪郭の点を描画する。
        ax.plot(cnt[:, 0], cnt[:, 1], "ro", mew=0, ms=4)
        # 輪郭の番号を描画する。
        ax.text(cnt[0][0], cnt[0][1], i, color="orange", size="20")





#外接矩形の座標計算
def make_boundingRect(sc2):
    contours, hierarchy = cv2.findContours(sc2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("{} contours.".format(len(contours)))

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_contours(ax, sc2, contours)
    plt.show()
    
    contours.sort(key=cv2.contourArea, reverse=True)
    
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    if len(contours) > 1:
        cnt_list = [cnt]
        x2 = x + w
        y2 = y + h
        
        for cnt_i in contours[1:]:
            if cv2.contourArea(cnt) * 0.2 <= cv2.contourArea(cnt_i):
                x_temp,y_temp,w_temp,h_temp = cv2.boundingRect(cnt_i)
                x2_temp = x_temp + w_temp
                y2_temp = y_temp + h_temp
            
                x = min(x,x_temp)
                y = min(y,y_temp)
                x2 = max(x2,x2_temp)
                y2 = max(y2,y2_temp)

        w = x2 - x
        h = y2 - y
    
    return x,y,w,h


# ## アノテーションデータの処理


#dataの読み込み
patient_num = 6
data_path = Path().cwd().resolve().parent/('Patient '+ str(patient_num))
output_dir = data_path.parent/'results'/'cutout'

if output_dir.exists() == False:
    output_dir.mkdir()

print('data_path:',data_path)
print('output_dir:',output_dir)

annotated_image_namelist = list(data_path.glob('annotatedImage*.jpg'))
image_namelist = list(data_path.glob('Image*.jpg'))

annotated_image_dic = make_image_dic(annotated_image_namelist)
annotated_image_list = list(annotated_image_dic.values())

image_dic = make_image_dic(image_namelist)
image_list = list(image_dic.values())



for img in tqdm(annotated_image_list):
    image_name = image_to_name(img,annotated_image_dic,False)
    image_name = image_to_name(img,annotated_image_dic,True)
    print('image:')
    img0 = annotated_to_image(img,annotated_image_dic,image_dic)#元画像
    
    dst,ct = thyroid_detect(img0) #エコー画像上でスキーマがある位置を探索
    sc, trim_range = thyroid_trim(img0,dst,ct) #エコー画像からスキーマの切り抜き
    cv2.imwrite(str(output_dir/('p'+str(patient_num)+image_name+'.png')), sc)



