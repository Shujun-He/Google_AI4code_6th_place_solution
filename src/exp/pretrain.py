import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
import sys, os
from metrics import *
import torch
import argparse
# from accelerate import Accelerator
# import accelerate
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
from logger import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/deberta-v3-large')
parser.add_argument('--train_mark_path', type=str, default='./data/train_mark.csv')
parser.add_argument('--train_features_path', type=str, default='./data/train_fts.json')
parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv')
parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json')
parser.add_argument('--val_path', type=str, default="./data/val.csv")
parser.add_argument('--load_weights_from', type=str, default="")

parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=768)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--preds_per_forward', type=int, default=5)
parser.add_argument('--tta', type=int, default=2)
parser.add_argument('--use_20_percent_data', action='store_true', help='only train with 20% data')
parser.add_argument('--val_only', action='store_true', help='only train with 20% data')
parser.add_argument('--local_rank', type=int, default=0)

torch.multiprocessing.set_sharing_strategy('file_system')

args = parser.parse_args()

#accelerator = Accelerator(accelerate.DistributedDataParallelKwargs(find_unused_parameters=True))
#device = accelerator.device
# model = MarkdownModel(args.model_name_or_path)
# model = model.to(device)

#exit()

# model = MarkdownModel(args.model_name_or_path)
# model = model.cuda()


#exit()

os.system("mkdir ./outputs")
#os.makedirs("outputs", mode = 0o777, exist_ok = True)
data_dir = Path('../../input/')
os.environ["TORCH_DISTRIBUTED_DEBUG"]="INFO" #0,1,2,3 for four gpu
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"
#CUDA_LAUNCH_BLOCKING=1


#exit()

train_df_mark = pd.read_csv(args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)

# lengths=[]
# for group in train_df_mark.groupby('id'):
#     lengths.append(len(group[1]))
# plt.subplot(1,2,1)
# plt.hist(lengths)
#exit()

if args.use_20_percent_data:
    train_df_mark=train_df_mark.iloc[:int(len(train_df_mark)*0.2)]

# lengths=[]
# for group in train_df_mark.groupby('id'):
#     lengths.append(len(group[1]))
# plt.subplot(1,2,2)
# plt.hist(lengths)
#
# plt.show()
#
# exit()

train_fts = json.load(open(args.train_features_path))
val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(args.val_features_path))
val_df = pd.read_csv(args.val_path)

#exit()

# val_df_mark=val_df_mark.iloc[:1000]
# val_df=val_df.iloc[:1000]

#exit()

order_df = pd.read_csv("../../input/train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

#exit()


class CustomCollate:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer

    def __call__(self,batch):
        ids, mask, fts, gather_ids, indices, labels =[],[],[],[],[],[]
        # print(len(batch))
        # exit()
        bs=len(batch)
        lengths=[]
        for i in range(bs):
            for inputs in batch[i]:
                lengths.append(len(inputs["segment_ids"]))

        max_len=max(lengths)

        sample_ids=[]
        code_pct_ranks=[]
        for i,data in enumerate(batch):
            for inputs in data:
                sample_len=len(inputs['segment_ids'])
                ids.append(torch.nn.functional.pad(inputs['segment_ids'],(0,max_len-sample_len),value=self.tokenizer.pad_token_id))
                mask.append(torch.nn.functional.pad(inputs['segment_mask'],(0,max_len-sample_len),value=0))
                gather_ids.append(torch.nn.functional.pad(inputs['segment_gather_ids'],(0,max_len-sample_len),value=0))
                #mask.append(data[1])
                fts.append(inputs['fts'])
                #gather_ids.append(data[3])
                sample_ids.append(i)

            indices.append(inputs['indices'])
            labels.append(inputs['labels'])
            #code_pct_ranks.append(inputs['code_pct_ranks']) #lx1-1xd == lxd .min(1)
            code_pct_ranks.append(inputs['code_pct_ranks']) #lx1-1xd == lxd .min(1)
            #code_pct_ranks.append(torch.abs(inputs['code_pct_ranks'].reshape(-1,1)-inputs['labels'].reshape(1,-1)).min(1)[0]) #lx1-1xd == lxd .min(1)
            # print("closest distance")
            # print(code_pct_ranks[-1])
            # print(len(code_pct_ranks[-1]))
            # print("code_pct_ranks")
            # print(inputs['code_pct_ranks'])
            # print(len(inputs['code_pct_ranks']))
            assert len(code_pct_ranks[-1])==len(inputs['code_pct_ranks'])
        #exit()
        ids=torch.stack(ids)#.cuda()
        mask=torch.stack(mask)#.cuda()
        fts=torch.stack(fts)#.cuda()
        gather_ids=torch.stack(gather_ids)#.cuda()
        indices=torch.cat(indices)
        labels=torch.cat(labels)#.cuda()
        sample_ids=torch.tensor(np.array(sample_ids))
        code_pct_ranks=torch.cat(code_pct_ranks)

        return ids, mask, fts, gather_ids, indices, labels, sample_ids, code_pct_ranks


if "deberta-v3" in args.model_name_or_path or "deberta-v2" in args.model_name_or_path:
    tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model_name_or_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

#train_ds[0]
train_ds = MarkdownPretrainDataset(train_df_mark, tokenizer=tokenizer, md_max_len=args.md_max_len,
                           total_max_len=args.total_max_len, fts=train_fts, preds_per_forward=args.preds_per_forward)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                          pin_memory=False, drop_last=True, collate_fn=CustomCollate(tokenizer))
val_ds = MarkdownPretrainDataset(val_df_mark, tokenizer=tokenizer, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts, preds_per_forward=args.preds_per_forward)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False, collate_fn=CustomCollate(tokenizer))


# Train a tokenizer
import os

os.environ["WANDB_DISABLED"] = "true"

import tokenizers
from transformers import BertTokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)


# In[2]:


import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from filelock import FileLock

class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer, file_path: str, block_size: int):
#         warnings.warn(
#             DEPRECATION_WARNING.format(
#                 "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
#             ),
#             FutureWarning,
#         )
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        #logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        self.lines=lines
        self.block_size=block_size

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        line=self.lines[i]
        batch_encoding = tokenizer(line, add_special_tokens=True, truncation=True, max_length=self.block_size)
        examples = batch_encoding["input_ids"]
        examples = {"input_ids": torch.tensor(examples, dtype=torch.long)}


        return examples


# In[3]:


from transformers import AutoModelWithLMHead, DataCollatorForLanguageModeling


model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

dataset= train_ds

# LineByLineTextDataset(
#     tokenizer = tokenizer,
#     file_path = './text.txt',
#     block_size = 64  # maximum sequence length
# )

print('No. of lines: ', len(dataset)) # No of lines in your datset


# In[ ]:


training_args = TrainingArguments(
    output_dir='./',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    save_steps=10000,
    dataloader_num_workers=4,
    fp16=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model('./')

torch.save(model.state_dict(),f"pytorch_model.bin")

# In[ ]:





# In[ ]:





# # Please upvote if you find it helpful! :D
