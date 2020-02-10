import os
import numpy as np
import csv
import pandas as pd
import torch
import torch.utils.data as data

cur_dir = os.path.dirname(os.path.abspath(__file__))
this_dir_name = cur_dir.split('/')[-1]
root_dir = os.path.join(cur_dir, '..')
img_folder=os.path.join(root_dir,'trainset','train.csv')

data = pd.read_csv(img_folder)
x1=data['image_path']
y1=data['labels']

for i,(x,y) in enumerate(zip(x1,y1)):
    print(x)
    print(y)
    if(i>=10):
        break



