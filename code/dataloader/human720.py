import os
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
from collections import deque
import cv2 as cv


class HumanDataLoader():
    def __init__(self, image_dir, label_path, pred_dir, stim_type, top1):

        self.image_dir = image_dir
        
        # FOR THE LABELS
        self.question_df = pd.read_csv(label_path)
        self.labels = list(self.question_df['Answer'].values)

        self.main_files = glob(pred_dir + '/main_test*.csv')
        if stim_type == 'opt':
            self.sel_ppl = list(range(300, 309)) + list(range(400, 408)) + [611] # 18 subjects
        elif stim_type == 'elec':
            self.sel_ppl = [499, 500, 502] + list(range(503, 509)) + list(range(602, 607)) + list(range(608, 612)) # 18 subjects
       
        # FOR THE HUMAN PREDICTIONS
        self.preds = deque()
        for m in tqdm(range(len(self.main_files))):
            temp_df = pd.read_csv(self.main_files[m])
            temp_df = temp_df[temp_df['유저식별아이디'].isin(self.sel_ppl)]
            if top1:
                temp_df = temp_df.loc[:, temp_df.columns.str.startswith('선택_A')]
            else:
                temp_df = temp_df.loc[:, temp_df.columns.str.startswith('선택_B')]
            for n in range(len(temp_df.columns)):
                temp_preds = temp_df.iloc[:, 0].values.astype(str).astype(float).astype(int)
                try:
                    temp_pred = np.bincount(temp_preds).argmax() 
                except:
                    temp_preds = [x for x in temp_preds if x > 0] # wierd value (-9223372036854775808)
                    temp_pred = np.bincount(temp_preds).argmax() 
                self.preds.append(temp_pred)

        assert len(self.labels) == len(self.preds)
    
    def __len__(self):
        return self.question_df.shape[0]
    
    def __getitem__(self, i): 
        label, pix, gs, par = self.labels[i].split('_')
        file_name = pix + '_' + gs + '_' + par
        pred = self.preds[i]

        temp_image_dir = os.path.join(self.image_dir, label)
        temp_image_dir = os.path.join(temp_image_dir, file_name)
        image = cv.imread(temp_image_dir, cv.IMREAD_GRAYSCALE)
        image = np.array(image.astype(np.float32))

        label = int(label)

        return image, label, pred
