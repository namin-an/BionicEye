import os
import random
import numpy as np
import pandas as pd
import itertools
import cv2 as cv


class KFaceDataLoader():
    def __init__(self, image_dir, data_path, label_path, class_num, train_test, seed):

        self.image_dir = image_dir
        
        df = pd.read_csv(data_path)
        l = list(range(df.shape[0]))
        random.seed(22)
        set_1 = random.sample(l, class_num)
        face_lst = [df.iloc[i, 0] for i in set_1]
        self.face_dic = {str(face) : int(i) for (i, face) in enumerate(face_lst)}

        if train_test == 'train':
            par_lst = os.listdir(os.path.join(self.image_dir, str(face_lst[0])))
            random.seed(seed)
            sel_par_lst = random.sample(par_lst, 1000) # Select 1k images per face
            self.datas = list(itertools.product(face_lst, sel_par_lst))
        
        elif train_test == 'test':
            question_df = pd.read_csv(label_path)
            self.datas = [(file_name.split('_')[0], file_name[len(file_name.split('_')[0])+1:]) for file_name in question_df['Answer'].values]
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, i): 
        label, par = self.datas[i][0], self.datas[i][1]

        temp_image_dir = os.path.join(self.image_dir, str(label))
        temp_image_dir = os.path.join(temp_image_dir, str(par))
        image = cv.imread(temp_image_dir, cv.IMREAD_GRAYSCALE)
        image = np.array(image.astype(np.float32))

        label = self.face_dic[str(label)]

        return image, label
