import csv
import os
from enum import Enum
from tqdm import tqdm

from statistics import mean, stdev

class XRAYTYPE(Enum):
    ELBOW='XR_ELBOW'
    FINGER='XR_FINGER'
    FOREARM='XR_FOREARM'
    HAND='XR_HAND'
    HUMERUS='XR_HUMERUS'
    SHOULDER='XR_SHOULDER'
    WRIST='XR_WRIST'

def is_equal_or_any(value, truth):
    '''
    Returns True if truth is None or if truth equals value.
    If truth is an enum, check the underlying value.
    Can be used to do optional equality checking.
    '''
    if isinstance(truth, XRAYTYPE):
        equal = value == truth.value
    else:
        equal = value == truth
    return truth is None or equal

def readcsv(file_name: str, x_ray_type: XRAYTYPE = None, patient:str = None, study:str = None):
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            split_path = row[0].split("/")
            if is_equal_or_any(split_path[2], x_ray_type) and is_equal_or_any(split_path[3], patient) and is_equal_or_any(split_path[4], study):
                yield row

def get_image_names_in_study(patient, study, image_path_csv, x_ray_type: XRAYTYPE = None):
    image_names = []
    for row in readcsv(image_path_csv, x_ray_type, patient, study):
        name = row[0].split('/')[-1]
        image_names.append(name)
    return image_names

def path_join(prefix, dirs):
    if len(dirs) == 0:
        return prefix
    else:
        prefix = os.path.join(prefix, dirs[0])
        dirs.pop(0)
        return path_join(prefix, dirs)

if __name__=="__main__":
    data_dir = os.path.join(os.getcwd(), 'data')
    dataset_type = 'train'
    # dataset_type = 'valid'
    label_file_path = os.path.join(data_dir, 'MURA-v1.1', dataset_type+'_labeled_studies.csv')
    image_file_path = os.path.join(data_dir, 'MURA-v1.1', dataset_type+'_image_paths.csv')
    lengths = []
    xray_type = XRAYTYPE.SHOULDER
    for row in tqdm(readcsv(label_file_path,xray_type)):
        split = row[0].split('/')
        patient = split[3]
        study = split[4]
        image_names = get_image_names_in_study(patient, study, image_file_path, xray_type)
        data = {
            'path': path_join(data_dir, split),
            'label': 1 if row[1]=='1' else -1,
            'img_names': image_names
        }
        if len(data['img_names']) < 1:
            print(data)
        lengths.append(len(data['img_names']))
        # print(data)
    print("Max: ", max(lengths))
    print("Mean: ", mean(lengths))
    print("StdDev: ", stdev(lengths))
    