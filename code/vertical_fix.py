'''
必要そうなものを取ってくる
'''
##縦は端があるかどうか判定するだけでok

#実際は left_edge_exist → up_edge_exist, right_edge_exist → bottom_edge_exist
def judge_edge_vertical(i0, img_thy_nod, edge_range=20):
    top,bottom,left,right = cut(i0)

    l_edge, r_edge = measure_col_thy(img_thy_nod)
    
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

    #左(left)の判定
    if left + edge_range <= l_edge:
        left_edge_exist = True
    
    elif s_cen[l_edge + 4] - 20 >= s_cen[l_edge + edge_range]:
        left_edge_exist = False
        print('left_state_1:',s_cen[l_edge + 4])
        print('left_state_2:',s_cen[l_edge + edge_range])
        
    elif np.mean(np.array(width_list[l_edge:l_edge+5])) <= 20:
        left_edge_exist  = True
        
    else:
        left_edge_exist = False
    

    #右(right)の判定
    if right - edge_range >= r_edge:
         right_edge_exist = True
    elif s_cen[r_edge - 4] - 20 >= s_cen[r_edge - edge_range]:
        right_edge_exist = False
        print('right_state_1:',s_cen[r_edge - 4])
        print('right_state_2:',s_cen[r_edge - edge_range])

    elif np.mean(np.array(width_list[r_edge-4:r_edge+1])) <= 20:
        right_edge_exist = True
    
    else:
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
        x = thy[:,cd]

        num = 0
        plot_list = []

        #上方向
        for i,y in enumerate(x):
            if y > 0 and num == 0:
                num += 1
                plot_list.append(i)
            elif y > 0 and num == 1:
                plot_list.append(i)
            
            elif y == 0 and num == 1:
                break 
        up_thy_edge = int(statistics.median(plot_list))
        up_thy_edge_list.append(up_thy_edge)

        up_thy_edge = edge_list_to_edge(up_thy_edge_list)

        num = 0
        plot_list = []
        #下方向
        for i in range(m):
            if x[m-1-i] > 0 and num == 0:
                num += 1
                plot_list.append(n-1-i)
            elif x[m-1-i] > 0 and num == 1:
                plot_list.append(n-1-i)
            
            elif x[m-1-i] == 0 and num == 1:
                break

        bottom_thy_edge = int(statistics.median(plot_list))
        bottom_thy_edge_list.append(bottom_thy_edge)

        bottom_thy_edge = edge_list_to_edge(bottom_thy_edge_list)

    return up_thy_edge, bottom_thy_edge



"""
手順
step1. 両方端がある場合:
up_thy_edge,bottom_thy_edgeに合わせる

step2.それ以外の場合:
プローブがシェーマ上の甲状腺からはみ出しているかどうかを基に判定を行う
step3. はみ出している場合
はみ出している分だけ移動する
step4.
端が存在しているがプローブの座標が甲状腺の端に位置していない場合は端に移動する

"""



"""
========
  main
========
"""

df_vertical = df_probe_fixed[df_probe_fixed['direction'] == 'vertical'].reset_index(drop=True)
thy_list =  list(df_probe_fixed['thy'])

if len(df_vertical) > 0:
    for i,image_name in enumerate(tqdm(df_vertical['image_name'].to_numpy().tolist())):
        print('===================')
        print(image_name)
        print('===================')
        img0 = annotated_image_dic[image_name]
        i0 = annotated_to_image(img0,annotated_image_dic,image_dic)
        print('original')
        imgplot(i0)

        print('annotated_image')
        imgplot(img0)
        #結節と甲状腺と悪性腫瘍のみを抽出した画像作成
        img_thy_nod = make_img_Thy_Nod_Mal(img0, threshold = 50)
        
        
        up_edge_exist,bottom_edge_exist = judge_edge_vertical(i0, img_thy_nod)
        print('up_edge_exist:',up_edge_exist)
        print('bottom_edge_exist:',bottom_edge_exist)
        

        #スキーマ
        img_thy = df_vertical['thy'][i]
        cd = df_vertical['cd'][i]
        th_min = df_vertical['th_min'][i]
        th_max = df_vertical['th_max'][i]

        print('th_min(pre):',th_min)
        print('th_max(pre):',th_max)
        imgplot(img_thy)

        up_thy_edge, bottom_thy_edge = calc_schema_edge_vertical(img_thy, thy_list, cd)

        #Step.1
        if up_edge_exist == True and bottom_edge_exist == True:
            th_max = bottom_thy_edge
            th_min = up_thy_edge
        else:
            #Step.2
            if up_thy_edge > th_min:
                th_max = th_max + (up_thy_edge - th_min)
                th_min =  up_thy_edge
            if bottom_thy_edge < th_max:
                th_min = th_min + (bottom_thy_edge - th_max)
                th_max = bottom_thy_edge

            #Step3
            if up_edge_exist == True and th_min != up_thy_edge:
                th_max = th_max + (up_thy_edge - th_min)
                th_min =  up_thy_edge

            elif bottom_edge_exist == True and th_max != bottom_thy_edge:
                th_min = th_min + (bottom_thy_edge - th_max)
                th_max = bottom_thy_edge
                
        print('cd:',cd)
        print('up_thy_edge:',up_thy_edge)
        print('bottom_thy_edge:',bottom_thy_edge)
        print('th_min:',th_min)
        print('th_max:',th_max)

        df_probe_fixed['th_min'].mask(df_probe_fixed['image_name'] == image_name, th_min ,inplace = True)
        df_probe_fixed['th_max'].mask(df_probe_fixed['image_name'] == image_name, th_max ,inplace = True)
