
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
pt = ['255','06']

img = cv2.imread('/test/Ito/SelectedP/'+pt[0]+'_Image0'+pt[1]+'.jpg')
t,b,l,r = cut(img)
img2 = cv2.imread('/test/Ito/test2/a_'+pt[0]+'_Image0'+pt[1]+'.jpg')
img2 = img2[t:b,l:r]

cdict = {}

#Thyroid
cdict['TL'] = np.array([50, 150, 0])# 抽出する色の下限(BGR)
cdict['TU'] = np.array([140, 240, 70])# 抽出する色の上限(BGR)

#Nodule
cdict['BL'] = np.array([0, 20, 100])
cdict['BU'] = np.array([60, 100, 250])

#Malignant
cdict['ML'] =  np.array([180, 80, 0])
cdict['MU'] = np.array([250, 120, 20])


#df_P = pd.DataFrame(columns = ['patient_name','type','no_of_images','df_I_list','schema'])

df_I = pd.DataFrame(columns = ['image_name','dimensions','shape','direction','side','df_US','sc2_dim','sh_dim','t_center','t_edge','s_edge'])
I1 = pd.Series([pt[0]+'_'+pt[1],(t,b,l,r),img2.shape[:2],None,None,None,None,None,None,None,None],index = df_I.columns)

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
        
    down = sm[np.where(np.array(sm)[:,1]==0)[0][0]][0] + 5
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
    
I1['direction'],I1['side'],I1['sc2_dim'],I1['sh_dim'],sc2,sh = process(img)

scol,slist,t_width,t_dist = calc_width(img2)

I1['t_center'] = detect_center(img2,scol,slist)

I1['t_edge'] = detect_edge(img2, slist, I1, edge_range=20)

I1['s_edge'] = calc_schema_edge(sc2,I1)

I1, sh = shift_probe(I1,sh,(slist[0,1],slist[-1,1]))

df_US = dist_thy_nod(I1,df_US)



'''
fig = plt.figure()
ax = fig.add_subplot(111)

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


'''