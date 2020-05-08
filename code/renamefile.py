from pathlib import Path
from pprint import pprint
import re

data_path = Path().cwd().resolve().parent

patient_path_list = data_path.glob('Patient [0-9]*')

for patient_path in patient_path_list:
    #print(patient_path)
    annotated_image_list = patient_path.glob('a_*_Image*.jpg') 
    image_list = patient_path.glob('[0-9]*_Image*.jpg') 
    
    for image_path in image_list:
        image_path.rename(patient_path/re.sub('[0-9]*_','',image_path.name))

    for annotated_path in annotated_image_list:
        annotated_path.rename(patient_path/re.sub('a_[0-9]*_','annotated',annotated_path.name))

#pprint.pprint(list(data_path.glob('Image*.jpg')))
#pprint.pprint(list(data_path.glob('a_*_Image*.jpg')))
#pprint.pprint(list(data_path.glob('[1-9]*_Image*.jpg')))

