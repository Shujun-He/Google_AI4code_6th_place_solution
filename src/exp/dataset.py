from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

import re

def preprocess_text(source):
        # Remove all the special characters
    source = re.sub(r'\W', ' ', str(source))
    #
    # # remove all single characters
    # document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    #
    # # Remove single characters from the start
    # document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    #
    # # Substituting multiple spaces with single space
    # document = re.sub(r'\s+', ' ', document, flags=re.I)
    #
    # # Removing prefixed 'b'
    source = re.sub(r'^b\s+', '', source)
    #
    # # Converting to Lowercase
    source = source.lower()
    #pattern = r'\<.*?\>'
    #document = re.sub(pattern, '', document)
    # source=source.split('\n')
    # new_source=''
    # for s in source:
    #     new_source+=s[:128]
    #     new_source+=' '
    return source


class MarkdownDataset(Dataset):

    def __init__(self, df, tokenizer, total_max_len, md_max_len, fts, preds_per_forward=5):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config

        self.tokenizer=tokenizer

        #self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.fts = fts
        self.preds_per_forward=preds_per_forward
        self.coordinates=[]
        self.df_chuncks={}
        for group in tqdm(df.groupby('id')):


        #for notebook_id in tqdm(df['id'].unique()):
            #notebook_df=df[df['id']==notebook_id]
            notebook_id=group[0]
            notebook_df=group[1]
            self.df_chuncks[notebook_id]=notebook_df
            #self.coordinates.append(notebook_df)
            n_preds=np.ceil(len(notebook_df)/self.preds_per_forward).astype('int')
            for i in range(n_preds):
                start=int(i*self.preds_per_forward)
                end=min(int((i+1)*self.preds_per_forward),len(notebook_df))
                self.coordinates.append({"coor":(start,end),"id":notebook_id})



        # print(df.iloc[0])
        # print(self.coordinates[0])
        # print(df.iloc[-1])
        # print(self.coordinates[-1])
        # print(len(self.coordinates))
        # print(len(df))
        # exit()

    def __getitem__(self, index):
        #row = self.df.iloc[index]

        #df_chunck=self.coordinates[idx]
        start,end=self.coordinates[index]['coor']
        notebook_id=self.coordinates[index]['id']
        df_chunck=self.df_chuncks[notebook_id].iloc[start:end]

        #row=df_chunck.iloc[0]

        # print(df_chunck)
        # exit()
        ids=[]
        gather_ids=[]
        mask=[]
        for i,source in enumerate(df_chunck['source']):
            inputs = self.tokenizer.encode_plus(
                preprocess_text(source),
                None,
                add_special_tokens=True,
                max_length=self.md_max_len,
                padding=False,
                return_token_type_ids=True,
                truncation=True
            )
            ids+=inputs['input_ids']
            ids+=[self.tokenizer.sep_token_id, ]
            gather_ids+=[i+1]*len(inputs['input_ids'])
            gather_ids+=[0]
            mask+=inputs['attention_mask']
            mask+=[1]


        code_inputs = self.tokenizer.batch_encode_plus(
            [preprocess_text(str(x)) for x in self.fts[notebook_id]["codes"]],
            add_special_tokens=True,
            max_length=23,
            padding=False,
            truncation=True
        )


        allowable_code_len=self.total_max_len-len(ids)

        code_len=0
        n_code=0
        start=0

        n_md = self.fts[notebook_id]["total_md"]
        n_code = self.fts[notebook_id]["total_md"]

        labels=list(df_chunck.pct_rank.values)
        indices=list(df_chunck.index.values)

        labels=torch.tensor(labels)
        indices=torch.tensor(indices)

        inputs=[]

        for i in range(len(code_inputs['input_ids'])):
            #print(i)
            #print(len(code_inputs))
            code_len+=len(code_inputs['input_ids'][i])
            if code_len>allowable_code_len or i==(len(code_inputs['input_ids'])-1):
                segment_ids=ids[:]
                segment_gather_ids=gather_ids[:]
                segment_mask=mask[:]
                #print(start,i)
                for j,x in enumerate(code_inputs['input_ids'][start:i+1]):
                    segment_ids.extend(x[:-1])
                    segment_gather_ids.extend(len(x[:-1])*[-j-1])
                    segment_mask.extend(len(x[:-1])*[1])

                if n_md + n_code == 0:
                    fts = torch.FloatTensor([0,0,0,0])
                else:
                    fts = torch.FloatTensor([start/len(code_inputs['input_ids']), n_md/128, n_code/128, n_md / (n_md + n_code)])


                segment_ids = torch.LongTensor(segment_ids[:self.total_max_len])
                segment_gather_ids=torch.LongTensor(segment_gather_ids[:self.total_max_len])
                segment_mask = torch.LongTensor(segment_mask[:self.total_max_len])
                code_pct_ranks = torch.tensor(np.arange(0,1,1/len(self.fts[notebook_id]["pct_ranks"]))).float()

                # print(code_pct_ranks)
                # print(segment_gather_ids.min())
                # exit()

                inputs.append({"segment_ids":segment_ids,"segment_gather_ids":segment_gather_ids,
                               "segment_mask":segment_mask,"fts":fts,"labels":labels,
                               "indices":indices,"code_pct_ranks": code_pct_ranks})

                start=i
                code_len=len(code_inputs['input_ids'][i])

            if len(inputs)>4:
                break



        return inputs

    def __len__(self):
        #return self.df.shape[0]
        return len(self.coordinates)



class MarkdownPretrainDataset(MarkdownDataset):

    def __getitem__(self, index):
        #row = self.df.iloc[index]

        #df_chunck=self.coordinates[idx]
        start,end=self.coordinates[index]['coor']
        notebook_id=self.coordinates[index]['id']
        df_chunck=self.df_chuncks[notebook_id].iloc[start:end]

        #row=df_chunck.iloc[0]

        # print(df_chunck)
        # exit()
        ids=[]
        gather_ids=[]
        mask=[]
        for i,source in enumerate(df_chunck['source']):
            inputs = self.tokenizer.encode_plus(
                preprocess_text(source),
                None,
                add_special_tokens=True,
                max_length=self.md_max_len,
                padding=False,
                return_token_type_ids=True,
                truncation=True
            )
            ids+=inputs['input_ids']
            ids+=[self.tokenizer.sep_token_id, ]
            gather_ids+=[i+1]*len(inputs['input_ids'])
            gather_ids+=[0]
            mask+=inputs['attention_mask']
            mask+=[1]


        code_inputs = self.tokenizer.batch_encode_plus(
            [preprocess_text(str(x)) for x in self.fts[notebook_id]["codes"]],
            add_special_tokens=True,
            max_length=23,
            padding=False,
            truncation=True
        )


        allowable_code_len=self.total_max_len-len(ids)

        code_len=0
        n_code=0
        start=0

        n_md = self.fts[notebook_id]["total_md"]
        n_code = self.fts[notebook_id]["total_md"]

        labels=list(df_chunck.pct_rank.values)
        indices=list(df_chunck.index.values)

        labels=torch.tensor(labels)
        indices=torch.tensor(indices)

        inputs=[]

        for i in range(len(code_inputs['input_ids'])):
            #print(i)
            #print(len(code_inputs))
            code_len+=len(code_inputs['input_ids'][i])
            if code_len>allowable_code_len or i==(len(code_inputs['input_ids'])-1):
                segment_ids=ids[:]
                segment_gather_ids=gather_ids[:]
                segment_mask=mask[:]
                #print(start,i)
                for j,x in enumerate(code_inputs['input_ids'][start:i+1]):
                    segment_ids.extend(x[:-1])
                    segment_gather_ids.extend(len(x[:-1])*[-j-1])
                    segment_mask.extend(len(x[:-1])*[1])

                if n_md + n_code == 0:
                    fts = torch.FloatTensor([0,0,0,0])
                else:
                    fts = torch.FloatTensor([start/len(code_inputs['input_ids']), n_md/128, n_code/128, n_md / (n_md + n_code)])


                segment_ids = torch.LongTensor(segment_ids[:self.total_max_len])
                segment_gather_ids=torch.LongTensor(segment_gather_ids[:self.total_max_len])
                segment_mask = torch.LongTensor(segment_mask[:self.total_max_len])
                code_pct_ranks = torch.tensor(np.arange(0,1,1/len(self.fts[notebook_id]["pct_ranks"]))).float()

                # print(code_pct_ranks)
                # print(segment_gather_ids.min())
                # exit()

                inputs.append({"segment_ids":segment_ids,"segment_gather_ids":segment_gather_ids,
                               "segment_mask":segment_mask,"fts":fts,"labels":labels,
                               "indices":indices,"code_pct_ranks": code_pct_ranks})

                start=i
                code_len=len(code_inputs['input_ids'][i])

            if len(inputs)>4:
                break


        examples = {"input_ids": inputs[0]["segment_ids"]}
        return examples
