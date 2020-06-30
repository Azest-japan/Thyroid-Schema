
from dataprep import displt,ldist,detect_color,decode2
from m_sc import process,detect_all,calc_width,detect_center,grayscale,imgplot
import pandas as pd

ct = {}
#Thyroid 
ct['TL'] = np.array([40, 150, 0])# 抽出する色の下限(BGR)
ct['TU'] = np.array([120, 240, 70])# 抽出する色の上限(BGR)

#Nodule
ct['ML'] = np.array([0, 20, 100])
ct['MU'] = np.array([60, 100, 255])

#Malignant
ct['BL'] =  np.array([180, 80, 0])
ct['BU'] = np.array([255, 120, 20])

pred = model.predict(xtst)

num = 20
iname = lb[-500+num]

img = decode2(pred[num])

pt= iname.split('_')
no = int(pt[0])

df_I = pd.DataFrame(columns = ['image_name','dimensions','shape','direction','side','df_US','sc2_dim','sh_dim','t_center','t_edge','s_edge'])
df_US = detect_all(img,pt,ct)

#pt= lb[-500+num].split('_')

if no<201:
    i3 = cv2.imread('/test/Ito/Selected1/' + iname)
else:
    i3 = cv2.imread('/test/Ito/SelectedP/' + iname)
    
I1 = pd.Series([pt[0]+'_'+pt[1],(-1,-1,-1,-1),img.shape[:2],None,None,None,None,None,-1,None,None],index = df_I.columns)
if type(i3) != type(None):
    t,b,l,r = cut(i3)
    I1 = pd.Series([pt[0]+'_'+pt[1],(t,b,l,r),img.shape[:2],None,None,None,None,None,-1,None,None],index = df_I.columns)
    I1['direction'],I1['side'],I1['sc2_dim'],I1['sh_dim'],sc2,sh = process(i3)
    

ximg = np.uint8(np.zeros((320,512,3)))
ximg[:,:,0] = np.uint8(xtst[num].reshape((320,512))*255)
ximg[:,:,1] = ximg[:,:,0]
ximg[:,:,2] = ximg[:,:,0]
print(ximg.shape)
imgplot(np.hstack((ximg,img,decode(ytst[num]))))
plt.show()

m = np.zeros(img[0:2].shape)
for part in ['T','B','M']:
    if 'T' not in part:
        imgb = grayscale(detect_color(img,ct[part+'L'],ct[part+'U']))
        
        contours, _ = cv2.findContours(imgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        
        if np.sum(m)>0:
            imgplot(m)
            m = np.zeros(imgb.shape)
            
        for i,cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area>80:

                print('part: ',part,'part number: ',i,'Area: ',np.round(area/(dist*dist),3))
                h = cv2.convexHull(cnt)
                cv2.drawContours(m,cnt,-1,255, 2,2)
                cv2.drawContours(m,[h],-1,150, 2,1)
                maxd,maxp,mcd,dp = ldist(h.reshape(h.shape[0],2))
                print('max dist: ',np.round(maxd/dist,3))
                print('max dist through center: ',np.round(mcd/dist,3))
                
                d2 = df_US[df_US.part==part].iloc[i]
                d2['area'] = np.round(area/(dist*dist),3)
                d2['max_dist'] = np.round(maxd/dist,2)
                d2['max_dist_center'] = np.round(mcd/dist,2)
                df_US.loc[d2.name] = d2
                
                cv2.line(m,(maxp[0][0],maxp[0][1]),(maxp[1][0],maxp[1][1]),150,1)
                cv2.line(m,(dp[0][0],dp[0][1]),(dp[1][0],dp[1][1]),250,1)
                point = np.argmax(h[:,0])
                m = cv2.putText(m, str(i), (h[point][0][0]+5,h[point][0][1]+5), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = 255, thickness = 2, lineType=cv2.LINE_AA)
                

            #cv2.imwrite('/test/Ito/'+pt[0]+'_'+pt[1]+'_'+part+str(np.round(np.random.rand(),3))+'.jpg',m2)
    else:
        
        imgb = grayscale(img)
        contours, _ = cv2.findContours(imgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        area = cv2.contourArea(contours[0])
        x,y,w,h = cv2.boundingRect(contours[0])
        x2 = x+w
        y2 = y+h
        print(x,x2,y,y2)
        
        m = np.zeros(imgb.shape)
        for i,cnt in enumerate(contours):
            xt,yt,wt,ht = cv2.boundingRect(cnt)
            a2 = cv2.contourArea(cnt)
            if a2>0.2*area:
                
                print(cnt.shape,'add')
                
                cv2.drawContours(m,cnt,-1,255, 2,2)
                h = cv2.convexHull(cnt)

                cv2.drawContours(m,[h],-1,150, 2,1)
                
                maxd,maxp,mcd,dp = ldist(h.reshape(h.shape[0],2))
                print('\npart: ',part,'part number: ',i,'Area: ',np.round(a2/(dist*dist),3))
                print('max dist: ',np.round(maxd/dist,3))
                print('max dist through center: ',np.round(mcd/dist,3))
                
                d2 = df_US[df_US.part==part].iloc[i]
                d2['area'] = np.round(cv2.contourArea(cnt)/(dist*dist),3)
                d2['max_dist'] = np.round(maxd/dist,2)
                d2['max_dist_center'] = np.round(mcd/dist,2)
                df_US.loc[d2.name] = d2
                
                cv2.line(m,(maxp[0][0],maxp[0][1]),(maxp[1][0],maxp[1][1]),150,1)
                cv2.line(m,(dp[0][0],dp[0][1]),(dp[1][0],dp[1][1]),250,1)
                m = cv2.putText(m, str(i), (dp[0][0]-10,dp[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = 255, thickness = 2, lineType=cv2.LINE_AA)
                
                x_temp,y_temp,w_temp,h_temp = cv2.boundingRect(cnt)
                x2_temp = x_temp + w_temp
                y2_temp = y_temp + h_temp

                x = min(x,x_temp)
                y = min(y,y_temp)
                x2 = max(x2,x2_temp)
                y2 = max(y2,y2_temp)
                #print('-',x,x2,y,y2)

        img2 = np.uint8(np.zeros(img.shape))
        img2[y:y2,x:x2]  = img[y:y2,x:x2] 
        img = img2

        scol,slist,t_width,t_dist = calc_width(img)
        cwidth = -1

        if I1['direction'] == 'horizontal' or I1['side'] == -1:
            I1['t_center'],cwidth,index = detect_center(img,scol,slist)

        if I1['t_center']>0:
            Al = np.where(grayscale(img)[:,:I1['t_center']]!=0)[0].shape[0]
            Ar = np.where(grayscale(img)[:,I1['t_center']:]!=0)[0].shape[0]
            Wl = np.max(slist[:index,3])
            Wr = np.max(slist[index:,3])
            mw = np.where(slist[:,3] == Wl)[0]
            print('l_width',Wl)
            
            for w in mw:
                m[slist[w][2]:slist[w][2]+slist[w][3],slist[w][1]:slist[w][1]+1] = 255

            mw = np.where(slist[:,3] == Wr)[0]
            print('r_width',Wr)
            
            for w in mw:
                m[slist[w][2]:slist[w][2]+slist[w][3],slist[w][1]:slist[w][1]+1] = 255
            print('l_area',Al)
            print('r_area',Ar)

            cv2.line(m,(I1['t_center'],0),(I1['t_center'],320),250,1)

        maxwidth = np.max(slist[:,3])
        if cwidth>0:
            print('center_width',cwidth)

        mw = np.where(slist[:,3] == maxwidth)[0]
        print('max_width',maxwidth)

        for w in mw:
            m[slist[w][2]:slist[w][2]+slist[w][3],slist[w][1]:slist[w][1]+1] = 255

        #cv2.imwrite('/test/Ito/'+pt[0]+'_'+pt[1]+'_'+'w '+str(np.round(np.random.rand(),3))+'.jpg',img2)
imgplot(m)
plt.show()


