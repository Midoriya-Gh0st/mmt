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
        dataset_path = os.path.join(dataset_path, data + '_data.pkl' if if_align else data + '_data_noalign.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text_bert'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        print("ck-keys:", dataset[split_type].keys())
        # self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        # dict_keys(['raw_text', 'audio', 'vision', 'id', 'text', 'text_bert',
        # 'audio_lengths', 'vision_lengths', 'annotations', 'classification_labels', 'regression_labels'])
        self.labels = torch.tensor(dataset[split_type]['regression_labels'].astype(np.float32)).cpu().detach()  # TODO: BERT
        # input(f"ck-labels... {self.labels.shape}")
        # print(self.labels)

        # Note: this is STILL an numpy array
        # self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        """ BERT-data """
        self.names = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
        self.a_lens = dataset[split_type]['audio_lengths'] if 'audio_lengths' in dataset[split_type].keys() else None
        self.v_lens = dataset[split_type]['vision_lengths'] if 'vision_lengths' in dataset[split_type].keys() else None

        self.data = data

        self.n_modalities = 3  # vision/ text/ audio

        # bert_text = torch.from_numpy(dataset[split_type]['text_bert'])
        # glove_text = torch.from_numpy(dataset[split_type]['text'])
        # print("[BERT]:", bert_text.shape, bert_text.dtype)
        # print("[GloVe]:", glove_text.shape, glove_text.dtype)
        # print("[self]:", self.text.shape)

        self.glove_text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.bert_text = torch.tensor(dataset[split_type]['text_bert'].astype(np.float32)).cpu().detach()

        print("[self-glove]:", self.glove_text.shape, self.glove_text.dtype)
        print("[self-bert]:", self.bert_text.shape, self.bert_text.dtype)
        self.text = self.glove_text
        # self.text = torch.bmm(self.bert_text, self.glove_text)
        # self.text = self.glove_text
        print("[x]:", self.text.shape, self.text.dtype)

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
        # META = (0, 0, 0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])

        if self.data == 'mosi':
            # META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
            META = (self.names[index], self.a_lens[index], self.v_lens[index])  # BERT
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)

        # name = self.meta[index]
        # print(f"[name]: {name}")

        # print("index:", index)
        # print(f"META: {type(META), len(META)} \n[META]", META, "<<<.")
        # print(f'self-meta:{len(self.names)} {self.names}')

        # input()

        return X, Y, META
