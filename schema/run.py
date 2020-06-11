
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import gc
import os
from tqdm.notebook import tqdm
from pathlib import Path
from dataprep import cut
if '/test/Ito/Code/' not in sys.path:
    sys.path.append('/test/Ito/Code/schema')
from m_sc import *



#pt = ['205','11']
pt = ['205','02']

img = cv2.imread('/test/Ito/test2/'+pt[0]+'_Image0'+pt[1]+'.jpg')
img2 = cv2.imread('/test/Ito/test2/a_'+pt[0]+'_Image0'+pt[1]+'.jpg')

t,b,l,r = cut(img)
img2 = img2[t:b,l:r]

#df_P = pd.DataFrame(columns = ['patient_name','type','no_of_images','df_I_list','schema'])

df_I = pd.DataFrame(columns = ['image_name','dimensions','shape','direction','side','df_US','df_schema'])
I1 = pd.Series([pt[0]+'_'+pt[1],(t,b,l,r),img2.shape[:2],None,None,None,None],index = df_I.columns)

df_US = detect_all(img2,pt,cdict)


#df_I = df_I.append(I1, ignore_index = True)
    
def process(img):

    dst,ct = thyroid_detect(img,pl=False)
    sc,_ = thyroid_trim(img,dst,ct,pl=False)
    sh,direction = probe_detect(sc,pl=False)


    cmax,cd = thyroid_measure(sc,sh,direction,pl=False)
    sc1 = clean_schema(sc,sh,direction,pl=False)
    sc1[sc1<140] = 0
    m_sc1,n_sc1 = sc1.shape
    argsort = np.flip(np.argsort(np.sum(sc1[:m_sc1,:],axis=0)))
    neck = [argsort[np.where(argsort<cd)[0][0]],argsort[np.where(argsort>cd)[0][0]]]
    
    sc1 = sc1[:,neck[0]+2:neck[1]-2]
    sh = sh[:,neck[0]+2:neck[1]-2]
    
    cmax = cmax - neck[0] - 2

    imgplot(sc1)

    w = sc1.shape[1]
    sm = []
    for i in range(int((sc1.shape[0]+0.5)/2)):
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

    cen_row_min,cen_row_max,sc2 = remove_neck(sc1,cmax,pl=False)
    x,y,w,h,sc2 = make_boundingRect(sc2)

    cmax = x+int((w+0.5)/2)

    xp,yp,wp,hp,sh = clean_probe(sh,direction)
    #shift
    #length = 5
    #sh = cv2.rectangle(np.zeros(sh.shape),(x,y-length),(x+w-1,y-length+h),color=250,thickness=-1)

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
    imgplot(sc2)
    imgplot(sc2[y:y+h,x:x+w])
    
    s_schema = pd.Series()
    s_schema['sc2_dim'] = (x,x+w,y,y+h)
    s_schema['sh_dim'] = (xp,xp+wp,yp,yp+hp)
    side='right'
    if int(xp+(wp)/2+0.5) < int(x+(w)/2+0.5):
        side = 'left'
    print(int(xp+(wp)/2+0.5),int(x+(w)/2+0.5))
    
    return direction, side, s_schema, sc2, sh
    
I1['direction'],I1['side'],s_schema,sc2,sh = process(img)


fig = plt.figure()
ax = fig.add_subplot(111)

cdict = {}

#Thyroid
cdict['TL'] = np.array([50, 150, 0])# 抽出する色の下限(BGR)
cdict['TU'] = np.array([140, 240, 70])# 抽出する色の上限(BGR)

#Nodule
cdict['BL'] = np.array([0, 20, 100])
cdict['BU'] = np.array([60, 100, 200])

#Malignant
cdict['ML'] =  np.array([140, 50, 0])
cdict['MU'] = np.array([250, 150, 60])



'''

sc2[cen_row_min:cen_row_max,cmax] = 250


imgplot(sc2[y:y+h,x:x+w])


if (x1<x and x2<x) or (x1>x and x2>x):
    sc2[y:y+h,x+int((w+0.5)/2):x+w] = cv2.flip(sc2[y:y+h,x:x+int((w+0.5)/2)], 1)


plt.plot(np.arange(len(d)),d)
 
l = np.argmin(d2)
r = np.argmax(d2)

for i in range(r-l+2):
    d[l+i] = d[l]+i*(d[r+1]-d[l])/(r-l+1)

# use cv2.line

#step1: 外接矩形のshapeを決定
#step1.1:shapeの最頻値(mode)を取る
if df_probe['(w,h)'].nunique()>1:
    if df_probe['(w,h)'].value_counts()[0]/len(df_probe)>=0.5:
        (w,h) = df_probe['(w,h)'].value_counts().index[0]
    else:
        print('error: 外接矩形の形が一致しません。それぞれの画像における矩形を確認してください。')
else:
    (w,h) = df_probe['(w,h)'].unique()[0]

    
#step1.2: shapeが違うものは、使っていない方のプローブを利用して縦方向を補正する

for i in range(len(df_probe)):
    x, y = df_probe['(x,y)'][i]
    w_temp, h_temp = df_probe['(w,h)'][i]
    cmax = df_probe['cmax'][i]
    cd = df_probe['cd'][i]
    th_min = df_probe['th_min'][i]
    th_max = df_probe['th_max'][i]
    direction = df_probe['direction'][i]
    img = df_probe['sc2'][i]
    
    if abs(w_temp - w)>2 or abs(h_temp - h)>2:
        print('error: {}の外接矩形の形がずれています。'.format(df_probe['image_name'][i]))
    
    elif abs(w_temp - w)== 0 and abs(h_temp - h) <= 2 and abs(h_temp - h) > 0:
        print(df_probe['image_name'][i])
        print("処理:外接矩形の縦方向補正")
        if direction == 'vertical':
            if cd < cmax:
                img = img[:,cmax:x+w+1]
                imgplot(img)
            
            else:
                img = img[:,x:cmax+1]
                imgplot(img)
        else:
            if cmax < (th_min+th_max)/2:
                img = img[:,x:cmax+1]
                imgplot(img)
            else:
                img = img[:,cmax:x+w+1]
                imgplot(img)
                
        x1,y1,w1,h1 = make_boundingRect(img)
        df_probe['(x,y)'][i] = (x,y1)
        df_probe['(w,h)'][i] = (w_temp,h1)
        print("補正後(w,h):",(w_temp,h1))
    
    
#step1.3: 正しいものについてcmaxと各頂点の距離を計算する
distance_list = []
for i in range(len(df_probe)):
    if (w,h) == df_probe['(w,h)'][i]:
        (x,y) = df_probe['(x,y)'][i]
        dis_l = df_probe['cmax'][i] - x
        dis_r = x + w - 1 - df_probe['cmax'][i]
        dis_u = df_probe['cen_min'][i] - y
        dis_b = y + h - 1 - df_probe['cen_max'][i]
        
        distance_list.append((dis_l,dis_r,dis_u,dis_b))
        
count = Counter(distance_list)
if len(df_probe)>2:
    if count.most_common(1)[0][1]/len(df_probe) > 0.5:
        dist_cmax =  count.most_common(1)[0][0]
    else:
        cmax_list = [df_probe['(x,y)'][i][0] + int(Decimal(str(w/2)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))for i in range(len(df_probe))]
        count_cmax = Counter(cmax_list)
        if count_cmax.most_common(1)[0][1]/len(df_probe) > 0.5:
            cmax =  count_cmax.most_common(1)[0][0]
            df_probe['cmax'] = cmax
            print('処理(全体):cmaxを外接矩形から補正')
else:
    cmax_temp = df_probe['(x,y)'][i][0] + int(Decimal(str(w/2)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
    if cmax_temp != df_probe['cmax'][0]:
        df_probe['cmax'] = cmax_temp
        print('処理(全体):cmaxを外接矩形から補正')     

        
#Step2: 画像のサイズを揃える
shape_list = list(df_probe['shape'])
m_list = []
n_list = []

for i in shape_list:
    a,b = i
    m_list.append(a)
    n_list.append(b)
    
m = int(statistics.median(m_list))
n = int(statistics.median(n_list))

if w % 2 == 0 and n % 2 == 1:
    n = n + 1
elif w % 2 == 1 and n % 2 == 0:
    n = n + 1

if h % 2 == 0 and m % 2 == 1:
    m = m + 1
elif h % 2 == 1 and m % 2 == 0:
    m = m + 1

df_probe_fixed = pd.DataFrame(columns = df_probe.columns)

#Step3: 外接矩形が画像の中心になるように移動する
#Step4: df_probeの座標を更新する
for i in range(len(df_probe)):
    (x,y) = df_probe['(x,y)'][i]
    sc2 = df_probe['sc2'][i] #scでよいか要確認
    direction = df_probe['direction'][i]
    result, up, bottom, left, right = resize_schema(sc2,x,y,w,h,m,n)
    
    a = left - x
    b = up - y
    
    cmax = df_probe['cmax'][i] + a
    cen_min = df_probe['cen_min'][i] + b
    cen_max = df_probe['cen_max'][i] + b
    if direction == 'vertical':
        cd = df_probe['cd'][i] + a
        th_min = df_probe['th_min'][i] + b
        th_max = df_probe['th_max'][i] + b
    else:
        cd = df_probe['cd'][i] + b
        th_min = df_probe['th_min'][i] + a
        th_max = df_probe['th_max'][i] + a
        
    image_name = df_probe['image_name'][i]
    shape = (m, n)
    (x, y) = (left, up)
    (w_temp, h_temp) = (right - left, bottom - up)
    
    df_temp = pd.Series([image_name,direction,shape,(x,y),(w_temp,h_temp), cmax, cen_min, cen_max, cd, th_min, th_max, result],index = df_probe.columns)
    df_probe_fixed = df_probe_fixed.append(df_temp, ignore_index = True)

    
#step5:cmax, cen_min, cen_maxについて統一
#これ以降cen_min,cen_maxは使用しない
if df_probe_fixed['cmax'].nunique()>1:
    if df_probe_fixed['cmax'].value_counts().iat[0]/len(df_probe_fixed)>0.5:
        cmax = df_probe_fixed['cmax'].value_counts().index[0]

if df_probe_fixed['cen_min'].nunique()>1:
    if df_probe_fixed['cen_min'].value_counts().iat[0]/len(df_probe_fixed)>0.5:
        cen_min = df_probe_fixed['cen_min'].value_counts().index[0]
        
if df_probe_fixed['cen_max'].nunique()>1:
    if df_probe_fixed['cen_max'].value_counts().iat[0]/len(df_probe_fixed)>0.5:
        cen_max = df_probe_fixed['cen_max'].value_counts().index[0]


for i in range(len(df_probe_fixed)):
    if df_probe_fixed['(w,h)'][i] == (w,h):
        df_probe_fixed['cmax'][i] = cmax
        df_probe_fixed['cen_min'][i] = cen_min
        df_probe_fixed['cen_max'][i] = cen_max        

df_probe_fixed = df_probe_fixed.rename(columns={'sc2':'thy'})


'''