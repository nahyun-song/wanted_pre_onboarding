'''스크립트 내 포함해야하는 함수
- set_device()
- custom_collate_fn()
'''

import os
import sys
import pandas as pd
import numpy as np 
import torch
import random

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
    return device

def custom_collate_fn(batch):
    input_list, target_list = [], []
    
    for _input, _target in batch:
        input_list.append(_input)
        target_list.append(_target)
    
    tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")
    tensorized_input = tokenizer_bert(
        input_list,
        add_special_tokens=True,
        padding="longest",  
        truncation=True,
        max_length=512,
        return_tensors='pt' 
    )
    
    tensorized_label = torch.tensor(target_list)
    
    return tensorized_input, tensorized_label



'''
포함해야하는 클래스
- CustomDataset
- CustomClassifier
'''

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertModel


class CustomDataset(Dataset):
    """
    - input_data: list of string
    - target_data: list of int
    """
    
    def __init__(self, input_data:list, target_data:list) -> None:
        self.X = input_data
        self.Y = target_data
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]



class CustomClassifier(nn.Module):

    def __init__(self, hidden_size: int, n_label: int):
        super(CustomClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("klue/bert-base")

        dropout_rate = 0.1
        linear_layer_hidden_size = 32

        self.classifier = nn.Sequential(
        nn.Linear(hidden_size, linear_layer_hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(linear_layer_hidden_size, n_label)
        )

    

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # BERT 모델의 마지막 레이어의 첫번재 토큰을 인덱싱
        last_hidden_states = outputs[0] # last hidden states (batch_size, sequence_len, hidden_size)
        cls_token_last_hidden_states = last_hidden_states[:,0,:] # (batch_size, first_token, hidden_size)

        logits = self.classifier(cls_token_last_hidden_states)

        return logits