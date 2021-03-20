import csv
import os
from enum import Enum
from statistics import mean, stdev

import cv2 as cv

class XRAYTYPE(Enum):
    ELBOW='XR_ELBOW'
    FINGER='XR_FINGER'
    FOREARM='XR_FOREARM'
    HAND='XR_HAND'
    HUMERUS='XR_HUMERUS'
    SHOULDER='XR_SHOULDER'
    WRIST='XR_WRIST'

def _is_equal_or_any(value, truth):
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

def _readcsv(file_name: str, x_ray_type: XRAYTYPE = None, patient:str = None, study:str = None):
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            split_path = row[0].split("/")
            if _is_equal_or_any(split_path[2], x_ray_type) and _is_equal_or_any(split_path[3], patient) and _is_equal_or_any(split_path[4], study):
                yield row

def _get_image_names_in_study(patient, study, image_path_csv, x_ray_type: XRAYTYPE = None):
    image_names = []
    for row in _readcsv(image_path_csv, x_ray_type, patient, study):
        name = row[0].split('/')[-1]
        image_names.append(name)
    return image_names

def _path_join(prefix, dirs):
    if len(dirs) == 0:
        return prefix
    else:
        prefix = os.path.join(prefix, dirs[0])
        dirs.pop(0)
        return _path_join(prefix, dirs)

def _get_study_image_paths(data_dir, label_file_path, image_file_path, xray_type=None):
    data = []
    for row in _readcsv(label_file_path, xray_type):
        split = row[0].split('/')
        patient = split[3]
        study = split[4]
        image_names = _get_image_names_in_study(patient, study, image_file_path, xray_type)
        yield {
            'path': _path_join(data_dir, split),
            'label': 1 if row[1]=='1' else 0,
            'img_names': image_names
        } 

def _get_images(data: dict):
    imgs = []
    for img_path in data['img_names']:
        img = cv.imread(os.path.join(data['path'], img_path), cv.IMREAD_GRAYSCALE)
        imgs.append(img)
    return imgs, data['label']

def _extract_images(data):
    for instance in data:
        yield _get_images(instance)

def _get_dirs(dataset_type:str = 'train', xray_type: XRAYTYPE = None):
    data_dir = os.path.join(os.getcwd(), 'data')
    label_file_path = os.path.join(data_dir, 'MURA-v1.1', dataset_type+'_labeled_studies.csv')
    image_file_path = os.path.join(data_dir, 'MURA-v1.1', dataset_type+'_image_paths.csv')
    return data_dir, label_file_path, image_file_path

def get_image_generator(dataset_type:str = 'train', xray_type: XRAYTYPE = None):
    data_dir, label_file_path, image_file_path = _get_dirs(dataset_type, xray_type)
    return _extract_images(_get_study_image_paths(data_dir, label_file_path, image_file_path, xray_type))
    
def _count_images_in_study(data):
    return len(data['img_names'])

def _get_max_images_per_study_per_dataset_type(dataset_type:str = 'train', xray_type: XRAYTYPE = None):
    data_dir, label_file_path, image_file_path = _get_dirs(dataset_type, xray_type)
    studies = list(_get_study_image_paths(data_dir, label_file_path, image_file_path, xray_type))
    return _count_images_in_study(max(studies, key=_count_images_in_study))

def get_max_images_per_study(xray_type: XRAYTYPE = None):
    train = _get_max_images_per_study_per_dataset_type('train', xray_type)
    valid = _get_max_images_per_study_per_dataset_type('valid', xray_type)
    return max(train, valid)

if __name__=="__main__":
    
    # dataset_type = 'train' 
    dataset_type = 'valid'
    xray_type = XRAYTYPE.FOREARM
    max_images = get_max_images_per_study(xray_type)
    ds = get_image_generator(dataset_type, xray_type)
    for data, label in ds:
        print(len(data), label)