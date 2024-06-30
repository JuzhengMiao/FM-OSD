import numpy as np
import csv
import os
import json

import torch

from utils_my import make_dir, visualize, tensor_to_scaler

import pandas as pd
from pathlib import Path

def radial(pt1, pt2, factor=[1, 1]):
    return sum(((i-j)*s)**2 for i, j,s  in zip(pt1, pt2, factor))**0.5

class Evaluater(object):
    def __init__(self, pred, gt, eval_radius=[], save_root = '', name = 'test', spacing = [0.1, 0.1], imgs = None, img_names = None):
        self.pred = np.array(pred)
        self.gt = np.array(gt)
        self.RE_list = list()
        self.num_landmark = self.gt.shape[1]

        self.recall_radius = eval_radius  # 2mm etc
        self.recall_rate = list()

        self.total_list = dict()

        self.save_root = save_root 

        self.name = name

        self.spacing_y = spacing[0]
        self.spacing_x = spacing[1]

        self.imgs = imgs
        self.img_names = img_names
        self.img_root = os.path.join(self.save_root, self.name + "_visual")

    def set_recall_radius(self, recall_radius):
        self.recall_radius = recall_radius

    def reset(self):
        self.RE_list.clear()
        self.total_list = list()

    def calculate(self): 
        diff = self.pred - self.gt
        diff = np.power(diff, 2).sum(axis = -1)
        diff = np.sqrt(diff)  
        if self.spacing_y == self.spacing_x:
            diff = diff*self.spacing_y
        self.RE_list = diff
        
        return None
    
    
    def save_img(self, img, preds, landmark_list, img_name): 
        image_pred = visualize(img, preds, landmark_list)
        
        image_pred.save(os.path.join(self.img_root, f'{img_name}_pred.png'))
    
    def save_img_all(self): 
        self.img_root = Path(self.img_root)
        self.img_root.mkdir(exist_ok=True, parents=True)

        for i in range(len(self.imgs)):
            img = self.imgs[i]
            preds = self.pred[i]
            landmark_list = self.gt[i]
            img_name = self.img_names[i][0].split('/')[-1].split('.')[0]
            self.save_img(img, preds, landmark_list, img_name)

    def save_preds(self, preds, runs_dir, id_str):
        inference_marks = {id:[int(preds[0][id]), \
            int(preds[1][id])] for id in range(self.num_landmark)}
        dir_pth = os.path.join(runs_dir, 'pseudo_labels')
        if not os.path.isdir(dir_pth): os.mkdir(dir_pth)
        with open('{0}/{1}.json'.format(dir_pth, id_str), 'w') as f:
            json.dump(inference_marks, f)

    def gen_latex(self):
        string_latex = f'{self.mre:.2f} & '
        for item in self.sdr:
            string_latex += f'{item:.2f} & '
        return string_latex

    def cal_metrics(self):
        # calculate MRE SDR
        temp = np.array(self.RE_list) 
        Mean_RE_channel = temp.mean(axis=0) 
       
        print("ALL MRE {}".format(Mean_RE_channel.mean()))
        self.mre = Mean_RE_channel.mean()

        self.sdr = []
        for radius in self.recall_radius:
            total = temp.size
            shot = (temp < radius).sum()
            print("ALL SDR {}mm  {}".format\
                                 (radius, shot * 100 / total))
            self.sdr.append(shot * 100 / total)
        
        write_csv = os.path.join(self.save_root, self.name + "_metric_abstract.csv")
        
        save = pd.concat([pd.DataFrame({'Mean_RE_perlandmark':Mean_RE_channel}), pd.DataFrame({'Mean_RE_alllandmark':[self.mre]}), pd.DataFrame({'SDR':self.recall_radius}), pd.DataFrame({'SDR-value':self.sdr})], axis=1)
        save.to_csv(write_csv, index=False, sep=',')  

    