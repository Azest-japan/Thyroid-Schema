#this file needs df_probe_fixed_Patient5.pickle
##共通のschemaとしてPatient5 image020を採用

import pandas as pd
import pickle
import cv2

df_probe_fixed = pd.read_pickle('df_probe_fixed_patient5.pickle')

temp_schema = df_probe_fixed['thy'][12]
ret,temp_schema = cv2.threshold(temp_schema, 20, 255, cv2.THRESH_BINARY)
#cv2.imwrite(str(data_path.parent/'schema.png'),temp_schema)

df_schema = df_probe_fixed.iloc[12]
df_schema['thy'] = temp_schema
df_schema = df_schema[['shape','(x,y)','(w,h)','cmax','cen_min','cen_max','thy']] 

schema_dict = df_schema.to_dict()

with open('schema_dict.pickle', 'wb') as f:
    pickle.dump(schema_dict, f)