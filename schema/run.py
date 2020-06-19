
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

cdict = {}
cs = grayscale(cv2.imread('/test/Ito/schema2.jpg'))  # schema2 (96, 100, 684, 636)
xs,ys,ws,hs = (96, 100, 684, 636)

#Thyroid 
cdict['TL'] = np.array([50, 150, 0])# 抽出する色の下限(BGR)
cdict['TU'] = np.array([140, 240, 70])# 抽出する色の上限(BGR)

#Nodule
cdict['BL'] = np.array([0, 20, 100])
cdict['BU'] = np.array([60, 100, 250])

#Malignant
cdict['ML'] =  np.array([180, 80, 0])
cdict['MU'] = np.array([255, 120, 20])

pdict = {}
pdict['B'] = (200,100,0)
pdict['M'] = (20,80,240)

#pt = ['205','11']

cs2 = np.uint8(np.ones((hs,ws,3))*255)
cs2[:,:,:] = cs[ys:ys+hs,xs:xs+ws].reshape((hs,ws,1))
cs2[cs2<120] = 0
cs2[cs2>120] = 255
intersection = []
for ino in ['02','05','07',10,11,12,13,14,15,17,19]:

    pt = ['206',str(ino)]

    img = cv2.imread('/test/Ito/Ischema/'+pt[0]+'_Image0'+pt[1]+'.jpg')
    t,b,l,r = cut(img)
    img2 = cv2.imread('/test/Ito/test2/a_'+pt[0]+'_Image0'+pt[1]+'.jpg')
    img2 = img2[t:b,l:r]

    df_I = pd.DataFrame(columns = ['image_name','dimensions','shape','direction','side','df_US','sc2_dim','sh_dim','t_center','t_edge','s_edge'])
    I1 = pd.Series([pt[0]+'_'+pt[1],(t,b,l,r),img2.shape[:2],None,None,None,None,None,-1,None,None],index = df_I.columns)

    df_US = detect_all(img2,pt,cdict)

    I1['direction'],I1['side'],I1['sc2_dim'],I1['sh_dim'],sc2,sh = process(img)

    scol,slist,t_width,t_dist = calc_width(img2)
    if I1['direction'] == 'horizontal':
        I1['t_center'] = detect_center(img2,scol,slist)

    I1['t_edge'] = detect_edge(img2, slist, I1, edge_range=20)
    I1['s_edge'] = calc_schema_edge(sc2,I1)
    print(I1)
    #I1, sh = shift_probe(I1,sh,(slist[0,1],slist[-1,1]))

    df_US = dist_thy_nod(I1,df_US)
    df_US,intersection = transfer_schema(I1,df_US,intersection)
    fig = plt.figure()
    print('haha')
    cs2 = plot_schema(cs2,df_US,pdict,I1)

cv2.imwrite('/test/Ito/sc206-.jpg',cs2)


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