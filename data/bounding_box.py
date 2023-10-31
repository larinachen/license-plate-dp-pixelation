import os
import json
import re
import xml.etree.ElementTree as xet
from glob import glob

# source code for parsing annotations is referenced from @chinnakotla on Kaggle: 
# https://www.kaggle.com/code/chinnakotla17/number-plate-recognition

data_dir = os.path.dirname(os.path.abspath(__file__))
global_path = glob(f"{data_dir}/annotations/*.xml")

carID_to_bounding_box = {}
for filename in global_path:
    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    # parse carID from filename
    pattern = r"Cars\d+"
    carID = re.findall(pattern, filename)[0]
    # store a map of carID to its license plate bounding box 
    carID_to_bounding_box[carID] = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

output_string = json.dumps(carID_to_bounding_box)
file_path = f"{data_dir}/bounding_box.json"
with open(file_path, 'w') as file:
    file.write(output_string)