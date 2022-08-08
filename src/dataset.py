import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        # print(dataset_path)
        # input()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))

        clip = 5
        length = dataset[split_type]['labels'].shape[0] // clip
        # input(length)

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()[:length]
        print(self.vision.shape)
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()[:length]
        self.audio = dataset[split_type]['audio'].astype(np.float32)[:length]
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()[:length]
        
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
        self.data = data
        self.n_modalities = 3  # vision/ text/ audio

        # IEMOCAP: emotion number
        # print(self.labels[0])
        self.x0 = 0
        self.x1 = 0
        self.x2 = 0
        self.x3 = 0
        # def get_pos(L):
        #     return torch.argmax(L[:, -1])
        # for label in self.labels:
        #     pos = get_pos(label)
        #     if pos == 0:
        #         self.x0 += 1
        #     elif pos == 1:
        #         self.x1 += 1
        #     elif pos == 2:
        #         self.x2 += 1
        #     elif pos == 3:
        #         self.x3 += 1
        # print(self.x0, self.x1, self.x2, self.x3)
        # input()

    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)

            pos = torch.argmax(Y)
            # if pos == 0:
            #     self.x0 += 1
            # elif pos == 1:
            #     self.x1 += 1
            # elif pos == 2:
            #     self.x2 += 1
            # elif pos == 3:
            #     self.x3 += 1
            #
            # print(self.x0, self.x1, self.x2, self.x3)

        return X, Y, META        

