
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
from scipy import stats
from math import floor
from decimal import Decimal, ROUND_HALF_UP
import pprint

from dataprep import cut

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
    if t==b or l == r:
        return -1,-1
    
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
        if len(col_num_list)!=0:
            cd = int(statistics.median(col_num_list))
        else:
            cd = cd = np.argmax(sh.sum(axis=0))
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
                    
        if len(row_num_list)!=0:
            cd = int(statistics.median(row_num_list))
        else:
            cd = np.argmax(sh.sum(axis=1))
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
    return xp,yp,wp-1,hp-1,sh


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

    #fig, ax = plt.subplots(figsize=(8, 8))
    #draw_contours(ax, sc2, contours)
    #plt.show()
     
    x2 = x + w
    y2 = y + h
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
    ab = []
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
        #print(i,a,b,x+a,cmax+b,sc2[i,x+a],sc2[i,cmax+b])
        #imgplot(sc2[i:i+1,:x2+1])
        ab.append([a,b,i])
            
    for i in range(y,y2):
        a,b,it = ab[i-y]
        #print(a,b,x+a,cmax,cmax+b,x2)
        #imgplot(sc2[i:i+1,:x2+1])
        if a!=0 and b == 0:
            sc2[i,x2-a-1] = sc2[i,x+a]
            if i>y:
                sc2[i,x2-ab[i-1-y][0]-1] = sc2[i,x+ab[i-1-y][0]]
                
        elif a==0 and b!=0:
            sc2[i,cmax-b] = sc2[i,cmax+b]
            if i>y:
                sc2[i,cmax-ab[i-1-y][1]] = sc2[i,cmax+ab[i-1-y][1]]
        '''       
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
        '''     
        #imgplot(sc2[i:i+1,:x2+1])
        #print('')
            
    print('sc2 - 1')
    imgplot(sc2)
    
    
    return x,y,w,h,sc2


# Process the above code
def process(img):

    dst,ct = thyroid_detect(img,pl=False)
    sc,_ = thyroid_trim(img,dst,ct,pl=False)
    if type(sc)==int:
        return -1,-1,-1,-1,-1,-1
    
    sh,direction = probe_detect(sc,pl=False)

    cmax,cd = thyroid_measure(sc,sh,direction,pl=False)
    sc1 = grayscale(sc)
    sc1[sh>0] = 0
    if np.max(sh)<=0:
        return -1,-1,-1,-1,-1,-1
    
    contours, _ = cv2.findContours(sh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    xp,yp,wp,hp = cv2.boundingRect(contours[0])
    if direction == 'horizontal':
        if len(contours)>1:
            xp1,yp1,wp1,hp1 = cv2.boundingRect(contours[1])
            sc1[yp:yp+hp,xp:xp1+wp1] = 0
        else:
            sc1[yp:yp+hp,xp:xp+wp] = 0
    else:
        if len(contours)>1:
            xp1,yp1,wp1,hp1 = cv2.boundingRect(contours[1])
            sc1[yp:yp1+hp1,xp:xp+wp] = 0
        else:
            sc1[yp:yp+hp,xp:xp+wp] = 0
    
    sh2 = np.uint8(np.zeros(sh.shape))
    sh2[yp:yp+hp,xp:xp+wp] = sh[yp:yp+hp,xp:xp+wp]
    sh = sh2
    
    sc1[sc1<140] = 0
    m_sc1,n_sc1 = sc1.shape
    argsort = np.flip(np.argsort(np.sum(sc1[:m_sc1,:],axis=0)))
    neck = [argsort[np.where(argsort<cd)[0][0]],argsort[np.where(argsort>cd)[0][0]]]
    
    sc1 = sc1[:,neck[0]+2:neck[1]-2]
    sh = sh[:,neck[0]+2:neck[1]-2]
    
    cmax = cmax - neck[0] - 2

    #imgplot(sc1)

    w = sc1.shape[1]
    
    # Up
    sm = []
    for i in range(int((sc1.shape[0])/2+0.5)):
        s = 0
        for j in range(w):
            yd = int(32*j*(w-j)/(w*w) + i + 0.5)
            s += sc1[yd,j]
        sm.append(s)

    sm2 = np.where(np.array(sm)==0)[0]
    up = 0
    for i in range(2,len(sm2)):
        print(sm2[i]-sm2[i-1],sm2[i-1] - sm2[i-2])
        if sm2[i]-sm2[i-1] > 2*(sm2[i-1] - sm2[i-2]):
            up = sm2[i]+8
            print(up)
            break

    sc1[:up] = 0

    # Down
    sm = []
    for i in range(int(sc1.shape[0]/2),sc1.shape[0]-4):
        s = 0
        for j in range(w):
            yd = int(16*j*(w-j)/(w*w) + i + 0.5)
            s += sc1[yd,j]
        sm.append([i,s])
        
    temp = np.where(np.array(sm)[:,1]==0)[0]
    if len(temp)!=0:
        down = sm[temp[0]][0] + 5
        sc1[down:] = 0

    x,y,w,h,sc2 = make_boundingRect(sc1)

    cmax = x+int((w+0.5)/2)

    xp,yp,wp,hp,sh = clean_probe(sh,direction)
    #shift
    #length = 5
    #sh = cv2.rectangle(np.zeros(sh.shape),(x,y-length),(x+w-1,y-length+h),color=250,thickness=-1)
    
    '''
    d = []
    d2 = []
    for i in range(w):
        s = np.where(sc2[:,x+i]>0)
        d.append((np.max(s),np.min(s),i))
        if len(d)>1:
            d2.append([d[i][0] - d[i-1][0], d[i][1] - d[i-1][1]])
            #print(i,d[i],d2[i-1])

    diff = np.max(d2,axis=0)[0:2] - np.min(d2,axis=0)[0:2]
    print(diff)
    if np.max(diff) > 32:
        
        # Multiple gaps not possible
        
        c = np.argmax(diff)
        if np.argmax(np.array(d2)[:,c]) < np.argmin(np.array(d2)[:,c]):
            y1,x1 = d[np.argmax(np.array(d2)[:,c])][c],x+d[np.argmax(np.array(d2)[:,c])][2]
            y2,x2 = d[np.argmin(np.array(d2)[:,c])+1][c],x+d[np.argmin(np.array(d2)[:,c])+1][2]
        else:
            y1,x1 = d[np.argmax(np.array(d2)[:,c])+1][c],x+d[np.argmax(np.array(d2)[:,c])+1][2]
            y2,x2 = d[np.argmin(np.array(d2)[:,c])][c],x+d[np.argmin(np.array(d2)[:,c])][2]

        if np.sqrt((y1-y2)*(y1-y2)+(x1-x2)*(x1-x2))<8:
            sc2 = cv2.line(sc2,(x1,y1),(x2,y2),255,1)
    
    '''
    #imgplot(sc2)
    #imgplot(sc2[y:y+h,x:x+w])

    sc = grayscale(cv2.imread('/test/Ito/schema.png'))
    contours, _ = cv2.findContours(sc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    x2,y2,w2,h2 = cv2.boundingRect(cnt)
    
    sc = sc[y2:y2+h2,x2:x2+w2]
    sc = cv2.resize(sc, (w,h),interpolation = cv2.INTER_AREA)
    sc2[y:y+h,x:x+w] = sc
    sc2[sc2<100] = 0
    
    side='right'
    
    if int(xp+(wp)/2+0.5) < int(x+(w)/2+0.5):
        side = 'left'
    print(int(xp+(wp)/2+0.5),int(x+(w)/2+0.5))
    
    return direction, side, (x,x+w,y,y+h), (xp,xp+wp,yp,yp+hp) , sc2, sh


'''
Annotation
'''

#アノテーションデータの処理
#色を基に領域を抽出
def detect_color(img,bgrLower,bgrUpper):
    img_mask = cv2.inRange(img, bgrLower, bgrUpper) # BGRからマスクを作成
    result = cv2.bitwise_and(img, img, mask=img_mask)
    return result



def detect_all(img,pt,cdict):
    df = pd.DataFrame(columns = ['image_name','part','part_number','area','max_dist','max_dist_center','col','sc_points'])
    h,w  = img.shape[:2]
    for part in ['T','B','M']:
        img2 = img
        if 'T' not in part:
            print(part)
            img2 = detect_color(img,cdict[part+'L'],cdict[part+'U'])

        img0 = grayscale(img2)
        imgplot(img0)
        img0[img0>60] = 200
        img0[img0<=60] = 0

        #imgplot(img0)
        
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
            if (area > 60 and part!='T') or (area>120 and part=='T'):
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

                df_temp = pd.Series([pt[0]+'_'+pt[1],part,i,None,None,None, (min_col, max_col),None],index = df.columns)
                df = df.append(df_temp, ignore_index = True)
            
    return df


#アノテーション画像から甲状腺の中心、端を特定してプローブの位置を補正する
#甲状腺全体における各列の幅を計算

def calc_width(img,factor=0.27):
    img = grayscale(img)
    
    img[img<60] = 0
    img[img!=0] = 240
    _,w = img.shape
    slist = []
    
    index = 0
    for i in range(w):
        k = img[:,i]
        k = np.where(k>0)[0]
        if len(k)>0:
            mn = np.min(k)
            width = np.max(k) - mn
            # index, column, dist, width
            slist.append([index,i,mn,width])
            #print(index,i,k,mn,width)
            index += 1
    
    slist = np.array(slist)
    n = np.unique(np.sort(slist[:,3]))
    m = np.unique(np.sort(slist[:,2]))
    
    
    t_dist = m[int(m.shape[0]*0.27+0.5)]
    t_width = n[int(n.shape[0]*0.27+0.5)]
    
    scol = np.intersect1d(slist[np.where(slist[:,3]<t_width)[0],1],slist[np.where(slist[:,2]<t_dist)[0],1])

    img[:,scol] = 255
    
    scol2 = []
    j = 0
    for i in scol:
        scol2.append([np.where(slist[:,1]==i)[0][0],i])
    
    scol = np.array(scol2)
    #imgplot(img)
    
    return scol,slist,t_width,t_dist


#甲状腺の中心座標の特定
# only for horizontal
#幅が最大の幅の30%以下の列のなかで一番皮膚に近いところを中心として判定
def detect_center(img,scol,slist):
    
    smin = 0
    min_col = -1
    cwidth = -1
    index = -1
    for i in scol:
        w = slist[i[0],3]
        d = slist[i[0],2]
        #print(i,w,d)
        #print(np.sort(slist[:i[0],3])[int(i[0]*0.2)])
        #print(np.sort(slist[i[0]:,3])[int((slist.shape[0]-i[0])*0.2)])
        #print(np.min(slist[np.where(slist[:,2]<=d)[0],3]))
        
        if i[0]>=10 and ((w < np.sort(slist[:i[0],3])[int(i[0]*0.2)] and w < np.sort(slist[i[0]:,3])[int((slist.shape[0]-i[0])*0.2)]) or (w < np.min(slist[np.where(slist[:,2]<=d)[0],3]) and (w < np.sort(slist[:i[0],3])[int(i[0]*0.3)] and w < np.sort(slist[i[0]:,3])[int((slist.shape[0]-i[0])*0.3)]))):
            #print(i,w,d,smin,2*w+d)
            if smin ==0:
                smin = 2*w+d
                min_col = i[1]
                cwidth = w
                index = i[0]
            elif smin>2*w + d:
                smin = 2*w+d
                min_col = i[1]
                cwidth = w
                index = i[0]
                #print(i,w,d)
    
    img = grayscale(img)
    img[:,min_col-1:min_col+1] = 255
    #imgplot(img)
    
    return min_col,cwidth,index


def calc_inter(intersection):
    inter_list=[]
    h_list = [k[1] for k in intersection if k[0] == 'h']
    v_list = [k[1] for k in intersection if k[0] == 'v']

    for i, (h_x1, h_x2, h_y1, h_y2) in enumerate(h_list):
        for j, (v_x1, v_x2, v_y1, v_y2) in enumerate(v_list):  
            if h_x1<v_x1 and h_x2>v_x2 and h_y1>v_y1 and h_y2<v_y2:
                inter_list.append((int((v_x1 + v_x2)/2), int((h_y1 + h_y2)/2)))
                
    return inter_list

def detect_edge(img, slist, I1, edge_range=20): 
    
    img = grayscale(img)
    img[img<60] = 0
    img[img!=0] = 240
    _,w = img.shape
    le = -1
    re = -1
    direction = I1['direction']
    shape = I1['shape']
    side = I1['side']
    if direction == 'horizontal':
        if side == 'left' and slist[0,1]>5:
            return slist[0,1],-1
        if side == 'right' and slist[-1,1]<img.shape[1]-5:
            return -1, slist[-1,1]
    
    if True:
        middle = slist[:,2] + 0.5*slist[:,3] # center of thyroid
        plt.plot(np.arange(middle.shape[0]),middle)
        if slist[0,1]>25:
            le = slist[0,1]
        else:
            if np.max(slist[:6,3])<0.4*np.max(slist[:,3]):
                le = slist[0,1]
                
                minc = np.argmin(middle[:80])
                print(minc)
                if minc!=0:
                    slope1, intercept1, r_1, p_value1, std_err1 = stats.linregress(np.arange(minc),middle[:minc])
                    c2 = min(minc,20)
                    slope2, intercept2, r_2, p_value2, std_err2 = stats.linregress(np.arange(2*c2),middle[minc-c2:minc+c2])
                    if slope1 < 0 and r_1*r_1>0.92 and slope2>slope1 and r_2*r_2<0.92:
                        print('Left Trapezium')
                        le = -1
                else:
                    s = slist[:,2]
                    e = slist[:,2] + slist[:,3]
                    if (min(np.abs(e[10]-e[1]),np.abs(s[10] - s[1]))>10 or max(np.abs(e[10]-e[1]),np.abs(s[10] - s[1]))>20)  and (np.abs(e[10]-e[1])*2 < np.abs(s[10] - s[1]) or np.abs(e[10]-e[1]) > 2*np.abs(s[10] - s[1])):
                        le = -1
             
        #print(re)
        if slist[-1,1]<img.shape[1]-20 and np.max(slist[-6:,3])<0.48*np.max(slist[:,3]):
            re = slist[-1,1]
        else:
            if np.max(slist[-20:-10,3])<0.4*np.max(slist[:,3]):
                re = slist[-1,1]
                
                tmax = 80 - np.argmax(middle[-80:])
                minc = 80+tmax-np.min(np.where(middle[-80-tmax:-tmax]>(np.mean(middle[-80-tmax:-tmax])))) + 1
                #print(minc,middle[-minc:])
                slope1, intercept1, r_1, p_value1, std_err1 = stats.linregress(np.arange(minc-tmax),middle[-minc:-tmax])
                c2 = min(minc,20)
                slope2, intercept2, r_2, p_value2, std_err2 = stats.linregress(np.arange(2*c2),middle[-minc-c2-1:-minc+c2-1])
                print(minc,tmax,slope1,slope2,r_1*r_1,r_2*r_2)
                if (slope1 > 0 and r_1*r_1>0.956) or (slope2<slope1 and r_1*r_1>0.92 and r_2*r_2<0.92):
                    print('Right Trapezium')
                    re = -1
                else:
                    s = slist[:,2]
                    e = slist[:,2] + slist[:,3]
                    if (min(np.abs(e[-10]-e[-1]),np.abs(s[-10] - s[-1]))>10 or max(np.abs(e[-10]-e[-1]),np.abs(s[-10] - s[-1]))>20)  and (np.abs(e[-10]-e[-1])*2 < np.abs(s[-10] - s[-1]) or np.abs(e[-10]-e[-1]) > 2*np.abs(s[-10] - s[-1])):
                        re = -1
                    
                    
    t_edge = (le, re)
    return t_edge



#スキーマのエッジの座標を計算
def calc_schema_edge(sc2,I1):
    
    direction = I1['direction']
    x,x2,y,y2 = I1['sc2_dim']
    xp,xp2,yp,yp2 = I1['sh_dim']
    edge = [-1,-1]
    
    if direction == 'vertical':
        yl = int((yp+yp2)/2+0.5)
        xl = int((xp+xp2)/2+0.5)
        print(yl,xl)
        for i in range(y2-y+1):

            if edge[0]<0 and sc2[yl-i,xl] > 160:
                edge[0] = yl-i
                
            if edge[1]<0 and sc2[yl+i,xl] > 160:
                edge[1] = yl+i
                
            if edge[0]>=0 and edge[1]>=0:
                break
        if edge[0]<0 and edge[1]>0: 
            yl = yp2
            xl = int((xp+xp2)/2+0.5)
            print(yl,xl)
            edge = [-1,-1]
            for i in range(y2-y+1):

                if edge[0]<0 and sc2[yl-i,xl] > 160:
                    edge[0] = yl-i

                if edge[1]<0 and sc2[yl+i,xl] > 160:
                    edge[1] = yl+i

                if edge[0]>=0 and edge[1]>=0:
                    break
        elif edge[0]>0 and edge[1]<0: 
            yl = yp1
            xl = int((xp+xp2)/2+0.5)
            print(yl,xl)
            edge = [-1,-1]
            for i in range(y2-y+1):

                if edge[0]<0 and sc2[yl-i,xl] > 160:
                    edge[0] = yl-i

                if edge[1]<0 and sc2[yl+i,xl] > 160:
                    edge[1] = yl+i

                if edge[0]>=0 and edge[1]>=0:
                    break
                    
    else:
        yl = int((yp+yp2)/2+0.5)
        xl = int((xp+xp2)/2+0.5)
        print(yl,xl)
        for i in range(x2-x+1):
                #print(i,xl,yl,xl-i,sc2[yl,xl-i],'  ',xl+i,sc2[yl,xl+i])
            if edge[0]<0 and sc2[yl,xl-i] > 160:
                edge[0] = xl-i

            if edge[1]<0 and sc2[yl,xl+i] >160:
                edge[1] = xl+i

            if edge[0]>=0 and edge[1]>=0:
                break
                
        if edge[0]<0 and edge[1]>0: 
            yl = int((yp+yp2)/2+0.5)
            xl = xp2
            edge = [-1,-1]
            for i in range(x2-x+1):
                #print(i,xl,yl,xl-i,sc2[yl,xl-i],'  ',xl+i,sc2[yl,xl+i])
                if edge[0]<0 and sc2[yl,xl-i] > 160:
                    edge[0] = xl-i

                if edge[1]<0 and sc2[yl,xl+i] >160:
                    edge[1] = xl+i

                if edge[0]>=0 and edge[1]>=0:
                    break
        elif edge[0]>0 and edge[1]<0: 
            yl = int((yp+yp2)/2+0.5)
            xl = xp1
            edge = [-1,-1]
            for i in range(x2-x+1):
                #print(i,xl,yl,xl-i,sc2[yl,xl-i],'  ',xl+i,sc2[yl,xl+i])
                if edge[0]<0 and sc2[yl,xl-i] > 160:
                    edge[0] = xl-i

                if edge[1]<0 and sc2[yl,xl+i] >160:
                    edge[1] = xl+i

                if edge[0]>=0 and edge[1]>=0:
                    break
    
    return edge


def shift_probe(I1,sh,tste):
    le,re = I1['t_edge']
    t_center = I1['t_center']
    _,n = I1['shape']
    
    xp,xp2,yp,yp2 = I1['sh_dim']
        
    if I1['direction'] == 'vertical':
        top,bot = I1['s_edge']
        shl = yp2 - yp  # probe length
        if le<0 and re>=0:
            print('right edge',bot,re,n,shl)
            print(bot,re,n,shl)
            end = int(bot + (n-re)*shl/n + 0.5)
            start = end - shl
            
        elif le>=0 and re<0:
            print('left edge',top,le,n,shl)
            start = int(top - le*shl/n + 0.5)
            end = start + shl
            
        elif le>=0 and re>=0:
            print('two edges',top,bot,le,re,n,shl)
            start = int(top - le*shl/n + 0.5)
            end = int(bot + (n-re)*shl/n + 0.5)
        
        else:
            start = yp
            end = yp2
            print('no edge')
            if yp2>bot:
                end = bot
                start = end - shl
            elif yp<top:
                start = top
                end = start + shl
            
        print('y',yp,yp2)
        imgplot(sh)
        print('se',start,end)
        sh = np.zeros(sh.shape)
        sh[start:end,xp:xp2] = 250
        I1['sh_dim'] = (xp,xp2,start,end)
        imgplot(sh)
        
    else:
        
        side = I1['side']
        l = xp2 - xp
        s0,s1 = I1['s_edge']
        cmax = int((I1['sc2_dim'][0] + I1['sc2_dim'][1])/2 + 0.5)
        t_edge = I1['t_edge']
        print(t_center,t_edge,s0,s1,cmax)
        start = -1
        end = -1
        # shift
        
        '''
        if t_center <0 and (t_edge[0]>0 or t_edge[1]>0):
            if side == 'left':
                a,_ = tste
                print(1,I1['direction'],side,s0,a,l,n)
                start = int(s0 - a*l/n + 0.5)
                end = start + l
                
            else:
                _,a = tste
                a = n-a
                print(1,I1['direction'],side,s1,a,l,n)
                end = int(s1 + a*l/n  + 0.5)
                start = end - l
        '''
        
        # shift
        if t_center>0 and (t_edge[0]<0 and t_edge[1]<0):
            if side == 'left':
                
                a = int(n/2 - t_center + 0.5)
                print(3,I1['direction'],side,s0,a,l,n)
                mid = cmax + a*l/n
                start = int(mid - l/2 + 0.5)
                start = max(s0,start)
                end = start + l
                I1['side'] = 'lmiddle'
                
                
            
            elif side == 'right':
                
                a = int(n/2 - t_center + 0.5)
                print(3,I1['direction'],side,s1,a,l,n)
                mid = cmax + a*l/n
                end = int(mid + l/2 + 0.5)
                end = min(s1,end)
                start = end - l
                I1['side'] = 'rmiddle'
        
        # shift or stretch
        elif t_center>0 and (t_edge[0]>0 or t_edge[1]>0):
            # stretch
            if not (t_edge[0]>0 and t_edge[1]>0):
                if side == 'left':
                    
                    a,te = tste
                    b = te - t_center
                    print(4,I1['direction'],side,s0,a,b,' ',l,n)
                    
                    end = int((cmax - b*s0/(n-a))/(1-b/(n-a))+0.5)
                    start = int((s0 - a*cmax/(n-b))/(1-a/(n-b)) +0.5)
                    
                elif side == 'right':
                    
                    ts,te = tste
                    a = n-te
                    b = ts - t_center
                    print(4,I1['direction'],side,s1,a,b,' ',l,n)
                    end = int((s1 - a*cmax/(n-b))/(1-a/(n-b))+0.5)
                    start = int((cmax - b*s1/(n-a))/(1-b/(n-a)) +0.5)
                    
            # shift
            else:
                print(5,I1['direction'],side)
                a = int(n/2 - t_center + 0.5)
                mid = cmax + a*l/n
                start = int(mid - l/2 + 0.5)
                end = int(mid + l/2 + 0.5)
                I1['side'] = 'middle'
                
        print(yp,yp2,xp,xp2)
        imgplot(sh)
        print(start,end)
        if start == -1 and end == -1:
            return I1,sh
        
        if end > sh.shape[1]:
             sh = np.zeros((sh.shape[0],end+1))
        elif start < 0:
            sh = np.zeros((sh.shape[0],sh.shape[1]-start+1))
            start = 0
        else:
            sh = np.zeros(sh.shape)
        sh[yp:yp2,start:end] = 250
        I1['sh_dim'] = (start,end,yp,yp2)
        imgplot(sh)
    
    return I1, sh


# Executed after probe correction, changes when there is a change in length of probe
        
#列番号から結節の相対距離を計算 (気道が見えない場合のみ)
def dist_thy_nod(I1,df_US):
    
    xp1,xp2,yp1,yp2 = I1['sh_dim']
    x1,x2,y1,y2 = I1['sc2_dim']
    
    for i in range(len(df_US)):
        if 'T' not in df_US.loc[i]['part']:
            
            minc,maxc = df_US.loc[i]['col'][0],df_US.loc[i]['col'][1]
            print(minc,maxc)
            
            if I1['direction'] == 'vertical':
                
                yn1,yn2 = int(yp1 + minc/I1['shape'][1]*(yp2-yp1) + 0.5), int(yp1 + maxc/I1['shape'][1]*(yp2-yp1) + 0.5)
                yn2 = max(yn2,yn1+2)
                xn1,xn2 = int(xp1 + (xp2-xp1)/4 + 0.5), int(xp1 + 3*(xp2-xp1)/4+0.5)
                print(yn1-y1,yn2-y1,xn1-x1,xn2-x1)
            else:
                
                xn1,xn2 = int(xp1 + minc/I1['shape'][1]*(xp2-xp1) + 0.5), int(xp1 + maxc/I1['shape'][1]*(xp2-xp1) + 0.5)
                xn2 = max(xn2,xn1+2)
                yn1,yn2 = int(yp1 + (yp2-yp1)/4 + 0.5), int(yp1 + 3*(yp2-yp1)/4+0.5)

            df_US.at[i,'sc_points'] = (xn1-x1,xn2-x1,yn1-y1,yn2-y1)
    
    return df_US


#座標に換算
def transfer_schema(I1,df_US,intersection,sm = (684,636)):
    
    x1,y1 = I1['sc2_dim'][1] - I1['sc2_dim'][0],I1['sc2_dim'][3] - I1['sc2_dim'][2]
    x2,y2 = sm
    for i in range(len(df_US)):
        if 'T' not in df_US.loc[i]['part']: 
            p1,p2,q1,q2 = df_US.loc[i]['sc_points']
            p1 = int(p1*x2/x1+0.5)
            p2 = max(p1+1,int(p2*x2/x1+0.5))
              
            q1 = int(q1*y2/y1+0.5)
            q2 = max(q1+1,int(q2*y2/y1+0.5))
            df_US.at[i,'sc_points'] =  p1,p2,q1,q2
            if I1['direction'] == 'vertical':
                x = int((p1+p2)/2)
                intersection.append([I1['direction'][0],(x-2,x+3,q1,q2)])
            else:
                y = int((q1+q2)/2)
                intersection.append([I1['direction'][0],(p1,p2,y-2,y+3)])
    return df_US,intersection
    
                  
def plot_schema(cs2,df_US,pdict,I1):
    
    for i in range(len(df_US)):
        if 'T' not in df_US.loc[i]['part']: 
            x1,x2,y1,y2 = df_US.loc[i]['sc_points']
            print(df_US.loc[i]['part'],x1,x2,y1,y2,cs2.shape)
            if I1['direction'] == 'vertical':
                x = int((x1+x2)/2)
                cs2[y1:y2,x-2:x+3] = pdict[df_US.loc[i]['part']]
                #cv2.line(cs2,(x1,y1),(x1,y2),pdict[df_US.loc[i]['part']],5)
            else:
                print(x1,x2,y1,y2,pdict[df_US.loc[i]['part']])
                y = int((y1+y2)/2)
                cs2[y-2:y+3,x1:x2] = pdict[df_US.loc[i]['part']]
                #cv2.line(cs2,(x1,y1),(x2,y1),pdict[df_US.loc[i]['part']],5)
    
    imgplot(cs2)
    return cs2


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














































