import os
import random
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
import cv2 as cv


class HumanDataLoader():
    def __init__(self, image_dir, label_path, pred_dir, stim_type, top1, is_test):

        self.image_dir = image_dir
        
        # FOR THE LABELS
        self.question_df = pd.read_csv(label_path)
        self.labels = list(self.question_df['Answer'].values)

        self.main_files = glob(pred_dir + '/main_test*.csv')
        if stim_type == 'opt':
            self.sel_ppl = list(range(300, 309)) + list(range(400, 408)) + [611] # 18 subjects
        elif stim_type == 'elec':
            self.sel_ppl = [499, 500] + list(range(502, 509)) + list(range(601, 607)) + list(range(608, 611)) # 18 subjects
        random.seed(42)
        test_l = random.sample(self.sel_ppl, len(self.sel_ppl)//2)
        if is_test:
            self.sel_ppl = test_l
        else:
            self.sel_ppl = [per for per in self.sel_ppl if not per in test_l]

        # FOR THE HUMAN PREDICTIONS
        self.preds = []
        n = 9
        for i in tqdm(range(1, 80*n+1, 80), leave=False):
            j = i+79 
            temp_df = pd.read_csv(os.path.join(pred_dir, f'main_test({i}_{j}).csv'))
            temp_df = temp_df[temp_df['유저식별아이디'].isin(self.sel_ppl)]
            if top1:
                temp_df = temp_df.loc[:, temp_df.columns.str.startswith('선택_A')]
            else:
                temp_df = temp_df.loc[:, temp_df.columns.str.startswith('선택_B')]
            temp_df = temp_df.fillna(0)
            for n in range(len(temp_df.columns)):
                temp_preds = temp_df.iloc[:, n].values.astype(str).astype(float).astype(int)
                temp_pred = np.bincount(temp_preds).argmax() 
                self.preds.append(temp_pred)
        
        if os.path.basename(label_path) != '211105_QAs_for_Set0_CNN_SVC_4classes_partial.csv':
            self.preds = [pred for (i, pred) in enumerate(self.preds) if int(i) in self.question_df['Trial'].values]
        
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
