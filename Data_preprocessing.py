import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.datasets import DatasetFolder
import torchvision
from tqdm.notebook import tqdm as tqdm
import torch.nn.functional as F
import os
from torchsummary import summary
import cv2
import dlib
import cv2
import imutils
from imutils import face_utils
import shutil

def lim(x = [], y = []):
    row_min = min(x)-5
    row_max = max(x)+5
    column_min = min(y)-5
    column_max = max(y)+5
    return row_min, row_max, column_min, column_max

## For LRW
root_dir = '/home/chisc/workspace/wuzhenrong/LRW_science_fair/LRW/lipread_mp4/'
root = os.listdir(root_dir)

for idx in root:
    if idx == 'CENTRAL':
        continue
    store_root_dir = f'/home/chisc/workspace/wuzhenrong/LRW_science_fair/LRW_image/partA/valid/{idx}'
#     print(store_root_dir)
    if not os.path.exists(store_root_dir):
        os.mkdir(store_root_dir)
    print(f'{idx} is preprocessing')
    files = os.listdir(os.path.join(root_dir, idx, 'val'))
    person_cnt = 0
    currentframe = 0
    for file in tqdm(files):
        try:
            if file.split('.')[1] == 'txt':
                continue
            cam = cv2.VideoCapture(os.path.join(root_dir, idx, 'val', file))
            # Used as counter variable
            count = 0
            currentframe = 0
            # checks whether frames were extracted
            success = 1
            detector = dlib.get_frontal_face_detector() # Returns the default face detector
            predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

            store_dir = os.path.join(store_root_dir, file.split('.')[0])
            if not os.path.exists(store_dir):
                os.mkdir(store_dir)
            while(True):
                ret,frame = cam.read()
                if ret:
                    face_rects = detector(frame, 0)
                    for i, f in enumerate(face_rects):
                        shape = predictor(frame, f)
                        arr_x = []
                        arr_y = []
                        cnt = 0
                        for p in shape.parts():
                            pt_pos = (p.x, p.y)
                            if cnt > 47 and cnt < 60:
                                arr_x.append(p.x)
                                arr_y.append(p.y)
                            cnt+=1
                        ri, rx, ci, cx = lim(arr_x, arr_y)
                        cropped = frame[ci: cx, ri: rx]
                        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    name = '/frame_' + str(currentframe) + '.jpg'
                    cv2.imwrite(store_dir + name, cropped)
                    currentframe += 1
                else:
                    break
            person_cnt += 1
            cam.release()
            cv2.destroyAllWindows()
        except:
            pass
