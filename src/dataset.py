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

max_images = 0

def get_study_image_paths(data_dir, label_file_path, image_file_path, xray_type=None):
    num_images = []
    data = []
    for row in readcsv(label_file_path, xray_type):
        split = row[0].split('/')
        patient = split[3]
        study = split[4]
        image_names = get_image_names_in_study(patient, study, image_file_path, xray_type)
        yield {
            'path': path_join(data_dir, split),
            'label': 1 if row[1]=='1' else -1,
            'img_names': image_names
        } 
        num_images.append(len(image_names))
    global max_images
    max_images = max(num_images)
    print(max(num_images),  mean(num_images), stdev(num_images))

def get_images(data: dict):
    imgs = []
    for img_path in data['img_names']:
        img = cv.imread(os.path.join(data['path'], img_path), cv.IMREAD_GRAYSCALE)
        imgs.append(img)
    return imgs, data['label']

def extract_images(data):
    for instance in data:
        yield get_images(instance)

def get_image_generator(dataset_type:str = 'train', xray_type: XRAYTYPE = None):
    data_dir = os.path.join(os.getcwd(), 'data')
    label_file_path = os.path.join(data_dir, 'MURA-v1.1', dataset_type+'_labeled_studies.csv')
    image_file_path = os.path.join(data_dir, 'MURA-v1.1', dataset_type+'_image_paths.csv')
    return extract_images(get_study_image_paths(data_dir, label_file_path, image_file_path, xray_type))
    
def fold():
    # ds_stats = {
    #     'train': {
    #         XRAYTYPE.ELBOW: {
    #             'max': 7,
    #             'mean': 2.8112884834663627,
    #             'stdev': 0.9368346433086429
    #         },
    #         XRAYTYPE.FINGER: {
    #             'max': ,
    #             'mean': ,
    #             'stdev': 
    #         },
    #         XRAYTYPE.FOREARM: {
    #             'max': ,
    #             'mean': ,
    #             'stdev': 
    #         },
    #         XRAYTYPE.HAND: {
    #             'max': ,
    #             'mean': ,
    #             'stdev': 
    #         },
    #         XRAYTYPE.HUMERUS: {
    #             'max': ,
    #             'mean': ,
    #             'stdev': 
    #         },
    #         XRAYTYPE.SHOULDER: {
    #             'max': ,
    #             'mean': ,
    #             'stdev': 
    #         },
    #         XRAYTYPE.WRIST: {
    #             'max': ,
    #             'mean': ,
    #             'stdev': 
    #         }
    #     },
    #     'valid': {
    #         XRAYTYPE.ELBOW: {
    #             'max': 8,
    #             'mean': 2.9430379746835444,
    #             'stdev': 1.1957657295964794
    #         },
    #         XRAYTYPE.FINGER: {
    #             'max': 5,
    #             'mean': 2.6342857142857143,
    #             'stdev': 0.8795057576095273
    #         },
    #         XRAYTYPE.FOREARM: {
    #             'max': 10,
    #             'mean': 2.263157894736842,
    #             'stdev': 0.903651991559848
    #         },
    #         XRAYTYPE.HAND: {
    #             'max': 5,
    #             'mean': 2.754491017964072,
    #             'stdev': 0.5752638563623733
    #         },
    #         XRAYTYPE.HUMERUS: {
    #             'max': 5,
    #             'mean': 2.1333333333333333,
    #             'stdev': 0.5436197090427292
    #         },
    #         XRAYTYPE.SHOULDER: {
    #             'max': 5,
    #             'mean': 2.902061855670103,
    #             'stdev': 1.0459375526434163
    #         },
    #         XRAYTYPE.WRIST: {
    #             'max': 5,
    #             'mean': 2.780590717299578,
    #             'stdev': 0.8650235553263267
    #         }
    #     }
    # }
    pass

if __name__=="__main__":
    
    dataset_type = 'train' 
    # dataset_type = 'valid'
    xray_type = XRAYTYPE.ELBOW
    print(max_images)
    ds = get_image_generator(dataset_type, xray_type)
    for data, label in ds:
        print(len(data), label)
    print(max_images)
    