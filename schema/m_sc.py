
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


#エコー画像からスキーマ部分の特定
def thyroid_detect(img,pl=True):
    img_gray = grayscale(img)
    imgplot(img_gray)

#edge
    c = cv2.Canny(img_gray,650,800)
    if pl==True:
        print('c')
        imgplot(c)
    
#フィルタ
    kernel = np.ones((50,50),np.float32)/500
    dst = cv2.filter2D(c,-1,kernel,borderType = cv2.BORDER_WRAP)
    if pl==True:
        print('dst')
        imgplot(dst)

#中心を特定
    ct = np.int32(np.mean(np.where(dst==np.max(dst)),axis=-1))
    print('ct:',ct)
    
    return dst,ct

#エコー画像からスキーマ部分の切り出し
def thyroid_trim(img,dst,ct,pl=True):
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
    
    if pl==True:
        print('sc:')
        imgplot(sc)
    
    sc0 = grayscale(sc)
    sc0[sc0<100] = 0
    sc[sc0==0] = 0
    
    if pl == True:
        print('sc0')
        imgplot(sc0)

    if pl==True:
        print('sc')
        imgplot(sc)
    
    return sc,[t,b,l,r]


#スキーマ→プローブの抽出
def probe_detect(sc,pl=True):
    m,n,d = sc.shape
    sh = np.uint8(np.zeros((sc.shape[0],sc.shape[1])))

    for i in range(m):
        for j in range(n):
            b,g,r = sc[i,j]
            if (120 <= r and 100 <= g and g <= 190 and b <=140 and g > b+10) or (r<210 and r<b-15 and r<g-10):
                sh[i,j] = 255
    
    if pl == True:
        print('image sh:')
        imgplot(sh)
    
    if np.max(sh.sum(axis=0)) > np.max(sh.sum(axis=1)):
        direction = 'vertical'
    else: direction = 'horizontal'
    print('direction',direction)
    return sh, direction


#スキーマの甲状腺の中心の列方向の座標，プローブの座標計算
def thyroid_measure(sc,sh,direction,pl=True):
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
    if pl==True:
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
    
    if pl==True:
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
        
    elif direction == 'horizontal':
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
    
    if pl == True:
        print('sc_copy')
        imgplot(sc_copy)
    return cmax, cd


#プローブの残りの座標計算
def probe_measure(sh,cd,direction,pl=True):
    sh = clean_image(sh)
    if direction== 'vertical':
        sh_hot = sh[:,cd]
    elif direction=='horizontal':
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
def clean_schema(sc,sh,direction,pl=True):
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
    
    if pl == True:
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
def remove_neck(sc1,cmax,pl=True):
    #上側の消去
    '''
    m_sc1,n_sc1 = sc1.shape
    
    n=0     #Horizontal
    neck=0  #vertical
    cen_row_max = 0
    cen_row_min = 0

    for (i, x) in enumerate(sc1[:,cmax]):

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
    #sc2[:neck+1,:]=0
    
    if pl==True:
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
    if pl==True:
        imgplot(sc2)
    '''
    
    #下の探索
    
    m_sc1,n_sc1 = sc1.shape
    sc2 = sc1.copy()
    ub_copy = sc1.copy()
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
    '''
    #ノイズを除去
    sc2 = np.where(sc2<10,0,sc2)
    for i in range(1,m_sc1-1):
            for j in range(1,n_sc1-1):
                if (sc2[i,j] > 0 and sc2[i,j-1] == 0 and sc2[i,j+1] == 0 and sc2[i-1,j-1] == 0
                and sc2[i-1,j+1] == 0 and sc2[i+1,j-1] == 0 and sc2[i+1,j+1] == 0):
                    sc2[i,j] = 0
    if pl == True:
        print('sc2')
        imgplot(sc2)
    '''
    return cen_row_min,cen_row_max,sc2



def clean_probe(sh,direction='vertical'):
    contours, _ = cv2.findContours(sh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("{} contours.".format(len(contours)))
    contours.sort(key=cv2.contourArea, reverse=True)
    xp,yp,wp,hp = cv2.boundingRect(contours[0])
    sh2 = np.zeros(sh.shape)
    sh2[yp:yp+hp,xp:xp+wp] = sh[yp:yp+hp,xp:xp+wp]
    sh = sh2

    sh2 = np.zeros(sh.shape)
    if direction == 'vertical':
        sm = np.sum(sh,axis=0)
        sh2[:,np.where(sm>0.3*np.max(sm))[0]] = sh[:,np.where(sm>0.3*np.max(sm))[0]]
    else:
        sm = np.sum(sh,axis=1)
        sh2[np.where(sm>0.3*np.max(sm))[0],:] = sh[np.where(sm>0.3*np.max(sm))[0],:]

    imgplot(sh2)
    sh = np.uint8(sh2)
    
    contours, _ = cv2.findContours(sh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xp,yp,wp,hp = cv2.boundingRect(contours[0])
    sh[yp:yp+hp,xp:xp+wp] = 250
    return xp,yp,wp,hp,sh


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
    sc2[sc2<100] = 0
    contours, _ = cv2.findContours(sc2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    cmax = x + int((w+0.5)/2)
    
    print("{} contours.".format(len(contours)))

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_contours(ax, sc2, contours)
    plt.show()
     
    
    if len(contours) > 1:
        cnt_list = [cnt]
        x2 = x + w
        y2 = y + h
        
        for cnt_i in contours[1:]:
            xt,yt,wt,ht = cv2.boundingRect(cnt_i)
            
            print(x,x+w,xt,xt+wt,' ',y,y+h,yt,yt+ht)
            if np.max(sc2[yt:yt+ht,xt:xt+wt])>180 and (not((x<xt and x+w>xt+wt) and (y<yt and y+h>yt+ht)) or (cv2.contourArea(cnt) * 0.12 <= cv2.contourArea(cnt_i) or yt+ht<y+h/2)):
                print(cnt_i.shape,'add')
              
            #if cv2.contourArea(cnt) * 0.2 <= cv2.contourArea(cnt_i):
                x_temp,y_temp,w_temp,h_temp = cv2.boundingRect(cnt_i)
                x2_temp = x_temp + w_temp
                y2_temp = y_temp + h_temp
            
                x = min(x,x_temp)
                y = min(y,y_temp)
                x2 = max(x2,x2_temp)
                y2 = max(y2,y2_temp)
            else:
                print(cnt_i.shape,'remove')
                sc2[yt:yt+ht,xt:xt+wt] = 0
            
        w = x2 - x
        h = y2 - y
        cmax = x + int((w+0.5)/2)
        print(y,y2,x,x2,w,h,cmax)
        print('sc2 - 0')
        imgplot(sc2[y:y2,x:x2])
        
        for i in range(y,y2):
            if np.nonzero(sc2[i,x:cmax])[0].shape[0] !=0:
                #print(i,sc2[i,x:cmax],np.nonzero(sc2[i,x:cmax]),np.min(np.nonzero(sc2[i,x:cmax])))
                a = np.min(np.nonzero(sc2[i,x:cmax]))
            else:
                a = 0
            if np.nonzero(sc2[i,cmax:x2])[0].shape[0] !=0:
                #print(i,sc2[i,cmax:x2],np.nonzero(sc2[i,cmax:x2]),np.max(np.nonzero(sc2[i,cmax:x2])))
                b = np.max(np.nonzero(sc2[i,cmax:x2]))
            else:
                b = 0
            print(i,a,b,x+a,cmax+b,sc2[i,x+a],sc2[i,cmax+b])
            imgplot(sc2[i:i+1,:x2+1])
            if a!=0 and b == 0:
                sc2[i,x2-a] = sc2[i,x+a]
            elif a==0 and b!=0:
                sc2[i,cmax-b] = sc2[i,cmax+b]
            else:
                #print(sc2[i,x+a],sc2[i,cmax+b])
                if sc2[i,x+a]<sc2[i,cmax+b]:
                    if a> x2-cmax-b:
                        sc2[i,x+x2-cmax-b] = sc2[i,cmax+b]
                    sc2[i,x+a] = sc2[i,cmax+b]
                else:
                    if a<x2-cmax-b:
                        sc2[i,x2-a] = sc2[i,x+a]
                    sc2[i,cmax+b] = sc2[i,x+a]
            #print('')
            
    print('sc2 - 1')
    imgplot(sc2)
    
    contours, _ = cv2.findContours(sc2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    x2 = x+w
    cmax = x + int((w+0.5)/2)
    if x2-cmax > cmax-x:
        x = 2*cmax - x2
    else:
        x2 = 2*cmax - x

    print(np.sum(sc2[y:y+h,x:cmax]),np.sum(sc2[y:y+h,cmax:x2]))
          
    if np.sum(sc2[y:y+h,cmax:x2]) < np.sum(sc2[y:y+h,x:cmax]):
        #imgplot(cv2.flip(sc2[y:y+h,x:cmax],1))
        sc2[y:y+h,cmax:x2] = cv2.flip(sc2[y:y+h,x:cmax],1)
    else:
        #imgplot(cv2.flip(sc2[y:y+h,cmax:x2],1))
        sc2[y:y+h,x:cmax] = cv2.flip(sc2[y:y+h,cmax:x2],1)
            
    print('sc2 - 2')
    imgplot(sc2)
    
    contours, _ = cv2.findContours(sc2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    return x,y,w,h,sc2



'''
Annotation
'''

#アノテーションデータの処理
#色を基に領域を抽出
def detect_color(img,bgrLower,bgrUpper):
    img_mask = cv2.inRange(img, bgrLower, bgrUpper) # BGRからマスクを作成
    result = cv2.bitwise_and(img, img, mask=img_mask)
    return result

#領域の最小列，最大列を計算
#甲状腺の場合
def measure_col_thy(img):
    if len(img.shape)>2:
        img = grayscale(img)
    h,w  = img.shape[:2]
    n_min = 0
    n_max = 0
    for i in range(w-1):
        if np.count_nonzero(img[:,i]> 0)>0 and n_min==0 and np.count_nonzero(img[:,i+1]> 0)> 0:
            n_min += 1
            min_col = i
        if np.count_nonzero(img[:,w-1-i]> 0)>0 and n_max==0 and np.count_nonzero(img[:,w-1-i-1]> 0)> 0:
            n_max += 1
            max_col = w-1-i
    return min_col, max_col

#甲状腺以外
def measure_col(img):#関数名検討
    img = grayscale(img)
    h,w  = img.shape
    n = 0
    for i in range(w):
        if np.count_nonzero(img[:,i]> 0)>0 and n==0 and np.count_nonzero(img[:,i+1]> 0)> 0:
            n += 1
            min_col = i
        elif img[:,i].sum() == 0 and n==1:
            n += 1
            max_col = i
    return min_col, max_col



def detect_all(img,pt,cdict):
    df = pd.DataFrame(columns = ['image_name','part','part_number','area','col','sc_points'])
    h,w  = img.shape[:2]
    for part in ['T','B','M']:
        if 'T' not in part:
            img = detect_color(img2,cdict[part+'L'],cdict[part+'U'])

        img0 = grayscale(img)
        img0[img0>50] = 200
        img0[img0<=50] = 0

        imgplot(img0)
        
        #edged = cv2.Canny(img, 30, 200)    # find edges of nodule
        contours, hierarchy = cv2.findContours(img0.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.sort(key=cv2.contourArea, reverse=True)
        nc = len(contours)
        n = 0
        nl = {}
        for i,cnt in enumerate(contours):
            if nc>1:
                area = cv2.contourArea(cnt)
            else:
                area = img0[img0!=0].shape[0]
            if area > 30:
                n += 1
                print(area)
                part_temp = np.zeros((img.shape[0],img.shape[1]))
                cv2.drawContours(part_temp, contours, i, 255, -1)

                p,q,r = cnt.shape
                cnt = cnt.reshape((p,r))
                x_list = [cnt[j][0] for j in range(p)]
                print('No:',i+1)
                #imgplot(part_temp)

                min_col = min(x_list)
                max_col = max(x_list)

                df_temp = pd.Series([pt[0]+'_'+pt[1],part,i+1,area, (min_col, max_col),None],index = df.columns)
                df = df.append(df_temp, ignore_index = True)
            
    return df




        
#列番号から結節の相対距離を計算 (気道が見えない場合のみ)
def dist_thy_nod(I1,df_US,s_schema):
    
    xp1,xp2,yp1,yp2 = s_schema['sh_dim']
    
    for i in range(len(df_US)):
        if 'T' not in df_US.loc[i]['part']:
            
            minc,maxc = df_US.loc[i]['min_col'],df_US.loc[i]['max_col']
            print(minc,maxc)
            
            if I1['direction'] == 'vertical':
                print('haha')
                yp2 = yp2-yp1
                yp1 = 0
                yn1,yn2 = int(yp1 + minc/I1['shape'][1]*(yp2-yp1) + 0.5), int(yp1 + maxc/I1['shape'][1]*(yp2-yp1) + 0.5)
                xn1,xn2 = int(xp1 + (xp2-xp1)/4 + 0.5), int(xp1 + 3*(xp2-xp1)/4+0.5)
                print(yn1,yn2,xn1,xn2)
            else:
                xp2 = xp2 - xp1
                xp1 = 0
                xn1,xn2 = int(xp1 + minc/I1['shape'][1]*(xp2-xp1) + 0.5), int(xp1 + maxc/I1['shape'][1]*(xp2-xp1) + 0.5)
                yn1,yn2 = int(yp1 + (yp2-yp1)/4 + 0.5), int(yp1 + 3*(yp2-yp1)/4+0.5)

            df_US.loc[i]['sc_points'] = (xn1,xn2,yn1,yn2)
    
    return df_US



#アノテーション画像から甲状腺の中心、端を特定してプローブの位置を補正する

#甲状腺(thy)と結節(Nod)を抽出し，結合する
def make_img_Thy_Nod_Mal(img,threshold = 20):

    img = grayscale(img)
    # 二値化(閾値を超えた画素を255にする。)
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    imgplot(img)
    
    return img



#外接矩形からの中心とプローブの位置関係から甲状腺の中心の方向を判定


#甲状腺全体における各列の幅を計算
def clac_width_thy(img_thy_nod):
    m,n = img_thy_nod.shape
    min_list = []
    width_list = []
    for i in range(n):
        img_col = img_thy_nod[:,i]
        if (img_col != 0).any():
            nzero_list = np.where(img_col != 0)[0].tolist()
            min_temp = min(nzero_list)
            max_temp = max(nzero_list)
        
            width = max_temp - min_temp
        else:
            width = 0
            min_temp = 0
        
        width_list.append(width)
        min_list.append(min_temp)
        
    return min_list, width_list


#甲状腺の中心があるか判定
def judge_center_edge(min_width,max_width,img_thy_nod):
    if min_width > max_width * 0.3:
        center_exist = False
    else:
        center_exist = True
    
    return center_exist


#甲状腺の中心座標の特定
#幅が最大の幅の30%以下の列のなかで一番皮膚に近いところを中心として判定
def detect_center_thy(width_list_judge, max_width, cen_rate = 0.3):
    width_list_judge = np.array(width_list_judge)
    cen_col_list = np.where((width_list_judge <= max_width * cen_rate) & (width_list_judge != 0))[0].tolist()
    for i in cen_col_list:
        row_temp = np.inf
        for i in cen_col_list:
            if row_temp > min_list[i]:
                row_temp = min_list[i]
                min_col = i
    if thy_LR == 'right':
        min_col = img0.shape[1] - 1 - min_col
    
    return min_col


#スキーマのエッジの座標を計算
def calc_schema_edge(img,thy_list,thy_LR,cd):
    thy_temp = [i for i in thy_list if (i != img).any()]

    edge_list = []
    m,n = img.shape

    #端点の座標計算
    for thy in thy_temp:

        x= thy[cd,:]
        
        plot_list = []
        num = 0
        
        if thy_LR == 'left':
            for i in range(n):
                if x[n-1-i] > 0 and num == 0:
                    num += 1
                    plot_list.append(n-1-i)
                elif x[n-1-i] > 0 and num == 1:
                    plot_list.append(n-1-i)
            
                elif x[n-1-i] == 0 and num == 1:
                    break
        else:
            for i,y in enumerate(x):
                if y > 0 and num == 0:
                    num += 1
                    plot_list.append(i)
                elif y > 0 and num == 1:
                    plot_list.append(i)
            
                elif y == 0 and num == 1:
                    break 
        edge = int(statistics.median(plot_list))
        edge_list.append(edge)

    if len(set(edge_list))>1:
        if Counter(edge_list).most_common(1)[0][0]/len(edge_list)>=0.5:
            edge = Counter(edge_list).most_common(1)[0][0]
        else:
            print('error: 各画像におけるプローブの位置が一致しません。それぞれの画像を確認してください。')
    else:
        edge = edge_list[0]
    #print('edge')
    #print(edge)
    #print('edge_list:',edge_list)
    return edge


def judge_edge(i0, img_thy_nod, thy_LR, edge_range=20): 
    top,bottom,left,right = cut(i0)
    l_edge,r_edge = measure_col_thy(img_thy_nod)
    
    print('left:{},right:{}'.format(left,right))
    print('l_edge:{},r_edge:{}'.format(l_edge,r_edge))
    
    s_cen = np.zeros(i0.shape[1])
    width_list= []
    for i in range(img_thy_nod.shape[1]):
        img_temp = img_thy_nod[:,i]
        if (img_temp != 0).any():
            nzero_list = np.where(img_temp != 0)[0].tolist()
            min_temp = min(nzero_list)
            max_temp = max(nzero_list)
            width = max_temp - min_temp
            width_list.append(int(width))
            s_cen[i] = (min_temp + max_temp) / 2
        else:
            width_list.append(0)
    
    if (thy_LR == 'right' and left + edge_range <= l_edge) or (thy_LR =='left' and right - edge_range >= r_edge):
        edge_exist = True
    
    elif (thy_LR == 'right' and s_cen[l_edge + 4] - 20 >= s_cen[l_edge + edge_range]) or (thy_LR == 'left' and s_cen[r_edge - 4] - 20 >= s_cen[r_edge - edge_range]):
        edge_exist = False
        print(s_cen[l_edge + 4])
        print(s_cen[l_edge + edge_range])
        
    elif (thy_LR == 'right' and np.mean(np.array(width_list[l_edge:l_edge+5])) <= 20) or(thy_LR == 'left' and np.mean(np.array(width_list[r_edge-4:r_edge+1])) <= 20):
        edge_exist  = True
        
    else:
        edge_exist = False
        
        
    return edge_exist


#実際は left_edge_exist → up_edge_exist, right_edge_exist → bottom_edge_exist
def judge_edge_vertical(i0, img_thy_nod, edge_range=40):
    top,bottom,left,right = cut(i0)

    l_edge, r_edge = measure_col_thy(img_thy_nod)
    
    print('left:{},right:{}'.format(left,right))
    print('l_edge:{},r_edge:{}'.format(l_edge,r_edge))
    #imgplot(i0[top:bottom,left:right])
    s_cen = np.zeros(i0.shape[1])

    width_list= []
    for i in range(img_thy_nod.shape[1]):
        img_temp = img_thy_nod[:,i]
        if (img_temp != 0).any():
            nzero_list = np.where(img_temp != 0)[0].tolist()
            min_temp = min(nzero_list)
            max_temp = max(nzero_list)
            width = max_temp - min_temp
            width_list.append(int(width))
            s_cen[i] = (min_temp + max_temp) / 2
        else:
            width_list.append(0)

    #print('left_state_1:',s_cen[l_edge + 4])
    #print('left_state_2:',s_cen[l_edge + 20])
    print(np.mean(np.array(width_list[l_edge:l_edge+5])))
    #左(left)の判定
    if left + edge_range <= l_edge:
        #print('a1')
        left_edge_exist = True
    
    elif s_cen[l_edge + 4] - 15 >= s_cen[l_edge + 20]:
        #print('a2')
        left_edge_exist = False
        print('left_state_1:',s_cen[l_edge + 4])
        print('left_state_2:',s_cen[l_edge + 20])
        
    elif np.mean(np.array(width_list[l_edge:l_edge+5])) <= 20:
        #print('a3')
        left_edge_exist  = True
        
    else:
        #print('a4')
        left_edge_exist = False
    
    print(np.mean(np.array(width_list[r_edge-4:r_edge+1])))
    #print('right_state_1:',s_cen[r_edge - 4])
    #print('right_state_2:',s_cen[r_edge - 20])
    #右(right)の判定
    if right - edge_range >= r_edge:
       # print('b1')
        right_edge_exist = True
    elif s_cen[r_edge - 4] - 15 >= s_cen[r_edge - 20]:
       # print('b2')
        right_edge_exist = False
        print('right_state_1:',s_cen[r_edge - 4])
        print('right_state_2:',s_cen[r_edge - 20])

    elif np.mean(np.array(width_list[r_edge-4:r_edge+1])) <= 20:
        #print('b3')
        right_edge_exist = True
    
    else:
       # print('b4')
        right_edge_exist = False


    return left_edge_exist, right_edge_exist


def edge_list_to_edge(edge_list):
    if len(set(edge_list)) > 1:
        if Counter(edge_list).most_common(1)[0][0]/len(edge_list)>=0.5:
            edge = Counter(edge_list).most_common(1)[0][0]
        else:
            print('error: 各画像におけるプローブの位置が一致しません。それぞれの画像を確認してください。')
    else:
        edge = edge_list[0]
    
    return edge


def calc_schema_edge_vertical(img,thy_list,cd):
    thy_temp = [i for i in thy_list if (i != img).any()]

    up_thy_edge_list = []    
    bottom_thy_edge_list = []

    m,n = img.shape

    #端点の座標計算
    for thy in thy_temp:
        #imgplot(thy)
        x = thy[:,cd]

        num = 0
        plot_list_up = []

        #上方向
        for i,y in enumerate(x):
            if y > 0 and num == 0:
                num += 1
                plot_list_up.append(i)
            elif y > 0 and num == 1:
                plot_list_up.append(i)
            
            elif y == 0 and num == 1:
                break 
                
        num = 0
        plot_list_bottom = []
        #下方向
        for i in range(m):
            if x[m-1-i] > 0 and num == 0:
                num += 1
                plot_list_bottom.append(m-1-i)
            elif x[m-1-i] > 0 and num == 1:
                plot_list_bottom.append(m-1-i)
            
            elif x[m-1-i] == 0 and num == 1:
                break
                
        if len(plot_list_up) > 0 and len(plot_list_bottom) > 0:
            up_thy_edge = int(statistics.median(plot_list_up))
            bottom_thy_edge = int(statistics.median(plot_list_bottom))
            
            if up_thy_edge != bottom_thy_edge:
                bottom_thy_edge_list.append(bottom_thy_edge)
                up_thy_edge_list.append(up_thy_edge)
        
    print(up_thy_edge_list)
    print(bottom_thy_edge_list)
    up_thy_edge = edge_list_to_edge(up_thy_edge_list)
    bottom_thy_edge = edge_list_to_edge(bottom_thy_edge_list)

    return up_thy_edge, bottom_thy_edge



#座標に換算

def recalc_for_plot(l_x, l_y, p_x, p_y, p_w, p_h, x, y, w, h):
    l_x = (l_x - p_x) / p_w * w + x
    l_y = (l_y - p_y) / p_h * h + y
    return l_x, l_y

#腫瘍の位置をschemaの座標に変換
def recalc_probe(df, p_x, p_y, p_w, p_h, x, y, w, h, part_name):
    min_list = []
    max_list = []

    for i in range(len(df)):
        cd = df['cd'][i]
        th_min = df['th_min'][i]
        th_max = df['th_max'][i]
        min_col = df['min_col'][i]
        max_col = df['max_col'][i]

        if df['direction'][i] == 'vertical':
            nod_min_y = th_min+(th_max-th_min)*min_col
            nod_max_y = th_min+(th_max-th_min)*max_col
            
            cd, nod_min_y = recalc_for_plot(cd, nod_min_y, p_x, p_y, p_w, p_h, x, y, w, h)
            _, nod_max_y = recalc_for_plot(cd, nod_max_y, p_x, p_y, p_w, p_h, x, y, w, h)
            cd = int(Decimal(str(cd)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
            nod_min_y = int(Decimal(str(nod_min_y)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
            nod_max_y = int(Decimal(str(nod_max_y)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) 
            nod_min = [nod_min_y, cd]
            nod_max = [nod_max_y, cd]
        else:
            nod_min_x = th_min+(th_max-th_min)*min_col
            nod_max_x = th_min+(th_max-th_min)*max_col
            nod_min_x, cd = recalc_for_plot(nod_min_x, cd, p_x, p_y, p_w, p_h, x, y, w, h)
            nod_max_x, _ = recalc_for_plot(nod_max_x, cd, p_x, p_y, p_w, p_h, x, y, w, h)
            cd = int(Decimal(str(cd)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
            nod_min_x = int(Decimal(str(nod_min_x)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
            nod_max_x = int(Decimal(str(nod_max_x)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) 
            
            nod_min = [cd, nod_min_x]
            nod_max = [cd, nod_max_x]
    
        min_list.append(nod_min)
        max_list.append(nod_max)

    if part_name == 'nodule':
        df['nod_min'] = pd.Series(min_list)
        df['nod_max'] = pd.Series(max_list)
    else:
        df['mal_min'] = pd.Series(min_list)
        df['mal_max'] = pd.Series(max_list)

    return df

def add_nod_id(df):
    image_name = df['image_name']
    part_number = str(df['part_number'])
    image_name = image_name.replace('annotatedImage','')
    image_name = str(int(image_name))
    if df['direction'] == 'vertical':
        direction = 'V'
    else:
        direction = 'H'
        
    part_id = image_name + direction + 'B' + part_number
    return part_id

def add_mal_id(df):
    image_name = df['image_name']
    part_number = str(df['part_number'])
    image_name = image_name.replace('annotatedImage','')
    image_name = str(int(image_name))
    if df['direction'] == 'vertical':
        direction = 'V'
    else:
        direction = 'H'
        
    part_id = image_name + direction + 'M' + part_number
    return part_id


#複数のエコー画像から結節のクラスター作成を行う

def judge_intersect(p1, p2, d1, p3, p4, d2):
    if d1 != d2:
        if d1 == 'vertical':
            h_range = list(range(p1[0], p2[0]+1))
            h = p3[0]
            w_range = list(range(p3[1], p4[1]+1))
            w = p1[1]
        else:
            h_range = range(p3[0], p4[0]+1)
            h = p1[0]
            w_range = list(range(p1[1], p2[1]+1))
            w = p3[1]
            
        return h in h_range and w in w_range
        
    elif d1 == 'vertical' and d2 == 'vertical':
        h1 = list(range(p1[0], p2[0]+1))
        h2 = list(range(p3[0], p4[0]+1))
        if p1[1] == p3[1] and len(set(h1)&set(h2)) > 0:
            return True
        else:
            return False

    else:
        h1 = list(range(p1[1], p2[1]+1))
        h2 = list(range(p3[1], p4[1]+1))
        if len(set(h1)&set(h2)) > 0 and p1[0] == p3[0]:
            return True
        else:
            return False
        
        
def calc_near(p1, p2, d1, p3, p4, d2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)
    if d1 != d2:
        if d1 == 'horizonal':
            w_range = list(range(p1[1], p2[1]+1))
            h = p1[0]
        
            w = p3[1]
            h_range = list(range(p3[0],p4[0]+1))
        
        else:
            w_range = list(range(p3[1], p4[1]+1))
            h = p3[0]
        
            w = p1[1]
            h_range = list(range(p1[0],p2[0]+1))
        
        if w in w_range:
            return (min(abs(min(h_range)-h),abs(max(h_range)-h)), 0)
        elif h in h_range:
            return (0, min(abs(min(w_range)-w),abs(min(w_range)-w)))
        else:
            dis_list = [p3 - p1, p4 - p1, p3 - p2, p4 - p2]
            dis = (100,100)
            for i in dis_list:
                if sum(dis)>sum(i):
                    dis = i
            return tuple(dis)
        
    elif d1 == 'vertical':
        dis_w = abs(p1[1] - p3[1])
        if p1[0] <= p3[0]:
            if p2[0] >= p3[0]:
                return (0,dis_w)
            else:
                return (p3[0]-p2[0],dis_w)
            
        else:
            if p1[0] <= p4[0]:
                return (0,dis_w)
            else:
                return (p1[0]-p4[0],dis_w)
    else:
        dis_h = abs(p1[0] - p3[0])
        if p1[1] <= p3[1]:
            if p2[1] >= p3[1]:
                return (dis_h, 0)
            else:
                return(dis_h,p3[1] - p2[1])
        else:
            p1[1] >= p3[1]
            if p1[1] <= p4[1]:
                return (dis_h, 0)
            else:
                return(dis_h, p1[1] - p4[1])

            
def judge_intersect_near(df):
    intersect_list = []
    near_list = []
    for i in range(len(df)):
        for j in range(i+1,len(df)):
            p1 = df['nod_min'][i]
            p2 = df['nod_max'][i]
            d1 = df['direction'][i]
        
            p3 = df['nod_min'][j]
            p4 = df['nod_max'][j]
            d2 = df['direction'][j]
        
            if judge_intersect(p1, p2, d1, p3, p4, d2) == True and df['image_name'][i] != df['image_name'][j]:
                intersect_list.append([i,j])
                #img_schema_copy = img_schema.copy()
                #cv2.line(img_schema_copy, tuple(list(reversed(p1))), tuple(list(reversed(p2))), 255)
                #cv2.line(img_schema_copy, tuple(list(reversed(p3))), tuple(list(reversed(p4))), 255)
                #imgplot(img_schema_copy)
        
            else:
                h_dis, w_dis = calc_near(p1, p2, d1, p3, p4, d2)
                if df['image_name'][i] != df['image_name'][j] and h_dis <= 2 and w_dis <= 1:
                    near_list.append([i,j]) 
                    #img_schema_copy = img_schema.copy()
                    #cv2.line(img_schema_copy, tuple(list(reversed(p1))), tuple(list(reversed(p2))), 255)
                    #cv2.line(img_schema_copy, tuple(list(reversed(p3))), tuple(list(reversed(p4))), 255)
                    #imgplot(img_schema_copy)
        
                elif df['image_name'][i] != df['image_name'][j] and h_dis <= 1 and w_dis <= 2:
                    near_list.append([i,j])
                    #img_schema_copy = img_schema.copy()
                    #cv2.line(img_schema_copy, tuple(list(reversed(p1))), tuple(list(reversed(p2))), 255)
                    #cv2.line(img_schema_copy, tuple(list(reversed(p3))), tuple(list(reversed(p4))), 255)
                    #imgplot(img_schema_copy)
    return intersect_list,near_list


#最初にintersect_listをもとに同じ結節のグループを作る
def make_cluster(df,intersect_list,near_list):
    cluster_dic = {}
    n_cluster = 0


    for i in range(len(df)):
        for j in intersect_list:
            a, b = j

            if a == i and a in cluster_dic:
                if b not in cluster_dic:
                    cluster_dic[b] = cluster_dic[a]
                elif cluster_dic[b] < cluster_dic[a]:
                    keys = [k for k, v in cluster_dic.items() if v == cluster_dic[a]]
                    for key in keys:
                        cluster_dic[key] = cluster_dic[b]
                
                elif cluster_dic[b] > cluster_dic[a]:
                    keys = [k for k, v in cluster_dic.items() if v == cluster_dic[b]]
                    for key in keys:
                        cluster_dic[key] = cluster_dic[a]
            
            elif a == i and a not in cluster_dic:
                if b in cluster_dic:
                    cluster_dic[a] = cluster_dic[b]
                else:
                    cluster_dic[a] = n_cluster + 1
                    cluster_dic[b] = cluster_dic[a]
                    
            elif b == i and b in cluster_dic:
                if a not in cluster_dic:
                    print('error')
                elif cluster_dic[b] < cluster_dic[a]:
                    keys = [k for k, v in cluster_dic.items() if v == cluster_dic[a]]
                    for key in keys:
                        cluster_dic[key] = cluster_dic[b]
                
                elif cluster_dic[b] > cluster_dic[a]:
                    keys = [k for k, v in cluster_dic.items() if v == cluster_dic[b]]
                    for key in keys:
                        cluster_dic[key] = cluster_dic[a]
            elif b == i and b not in cluster_dic:
                print('error')
                
                
            #クラスタ数の更新
            cluster_list = sorted(list(set(cluster_dic.values())))
            n_cluster = len(cluster_list)
            for index,cluster in enumerate(cluster_list):
                keys = [k for k, v in cluster_dic.items() if v == cluster]
                for key in keys:
                    cluster_dic[key] = index + 1

            
    #near_list
    for i in range(len(df_nodule_probe)):
        for j in near_list:
            a, b = j

            if a == i and a in cluster_dic:
                if b not in cluster_dic:
                    cluster_dic[b] = cluster_dic[a]
                elif cluster_dic[b] < cluster_dic[a]:
                    keys = [k for k, v in cluster_dic.items() if v == cluster_dic[a]]
                    for key in keys:
                        cluster_dic[key] = cluster_dic[b]
                
                elif cluster_dic[b] > cluster_dic[a]:
                    keys = [k for k, v in cluster_dic.items() if v == cluster_dic[b]]
                    for key in keys:
                        cluster_dic[key] = cluster_dic[a]
            
            elif a == i and a not in cluster_dic:
                if b in cluster_dic:
                    cluster_dic[a] = cluster_dic[b]
                else:
                    cluster_dic[a] = n_cluster + 1
                    cluster_dic[b] = cluster_dic[a]
                    
            elif b == i and b in cluster_dic:
                if a not in cluster_dic:
                    print('error')
                elif cluster_dic[b] < cluster_dic[a]:
                    keys = [k for k, v in cluster_dic.items() if v == cluster_dic[a]]
                    for key in keys:
                        cluster_dic[key] = cluster_dic[b]
                
                elif cluster_dic[b] > cluster_dic[a]:
                    keys = [k for k, v in cluster_dic.items() if v == cluster_dic[b]]
                    for key in keys:
                        cluster_dic[key] = cluster_dic[a]
            elif b == i and b not in cluster_dic:
                print('error')
                
                
        #クラスタ数の更新
            cluster_list = sorted(list(set(cluster_dic.values())))
            n_cluster = len(cluster_list)
            for index,cluster in enumerate(cluster_list):
                keys = [k for k, v in cluster_dic.items() if v == cluster]
                for key in keys:
                    cluster_dic[key] = index + 1  

    #cluster_dicに記載されていないものも含めてcluster_listに登録
    cluster_list = []
    for i in range(len(df_nodule_probe)):
        if i in cluster_dic:
            cluster_list.append(cluster_dic[i])
        else:
            n_cluster += 1
            cluster_dic[i] = n_cluster
            cluster_list.append(n_cluster)
            
    return cluster_list


def show_cluster(df, img_schema):
    color_list = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0),(255,0,255),(255,100,0),(100,255,0),(255,0,100),(100,0,255),(0,255,100),(0,100,255),(100,255,100),(255,100,100),(100,100,255)]
    
    img_schema_all = img_schema.copy()
    n_cluster = len(set(list(df['cluster'])))
    for i in range(1,n_cluster+1):
        df_temp = df[df['cluster']==i].reset_index()
        for j in range(len(df_temp)):
            #img_schema_copy= np.uint8(np.zeros((sc.shape[0],sc.shape[1],3)))
            nod_min = (df_temp['nod_min'][j][1], df_temp['nod_min'][j][0])
            nod_max = (df_temp['nod_max'][j][1], df_temp['nod_max'][j][0])
            #cv2.line(img_schema_copy, nod_min, nod_max, color_list[i-1])
            cv2.line(img_schema_all, nod_min, nod_max, color_list[i-1])
            #imgplot(img_schema_copy)
            #imgplot(img_schema_all)
    imgplot(img_schema_all)














































