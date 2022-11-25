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
from accelerate import Accelerator
import accelerate
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
from logger import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str, default='../../input/deberta-v3-base/')
parser.add_argument('--train_mark_path', type=str, default='./data/train_mark.csv')
parser.add_argument('--train_features_path', type=str, default='./data/train_fts.json')
parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv')
parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json')
parser.add_argument('--val_path', type=str, default="./data/val.csv")
parser.add_argument('--load_weights_from', type=str, default="")

parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--preds_per_forward', type=int, default=5)
parser.add_argument('--tta', type=int, default=2)
parser.add_argument('--use_20_percent_data', action='store_true', help='only train with 20% data')
parser.add_argument('--val_only', action='store_true', help='only train with 20% data')

torch.multiprocessing.set_sharing_strategy('file_system')

args = parser.parse_args()

accelerator = Accelerator(accelerate.DistributedDataParallelKwargs(find_unused_parameters=True))
device = accelerator.device
model = MarkdownModel(args.model_name_or_path)
model = model.to(device)

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
train_ds = MarkdownDataset(train_df_mark, tokenizer=tokenizer, md_max_len=args.md_max_len,
                           total_max_len=args.total_max_len, fts=train_fts, preds_per_forward=args.preds_per_forward)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                          pin_memory=False, drop_last=True, collate_fn=CustomCollate(tokenizer))
val_ds = MarkdownDataset(val_df_mark, tokenizer=tokenizer, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts, preds_per_forward=args.preds_per_forward)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False, collate_fn=CustomCollate(tokenizer))

#exit()

#exit()

# import matplotlib.pyplot as plt
# lengths=[]
# for batch in tqdm(train_loader):
#     lengths.append(batch[0].shape[1])

# plt.hist(lengths)
# plt.show()
# exit()

os.system("mkdir logs")
columns=['epoch','train_loss','val_score']
logger=CSVLogger(columns,f"logs/log.csv")


def validate(model, val_loader, accelerator):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    rearrange_indices=[]

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            ids, mask, fts, gather_ids, indices, target, sample_ids, code_pct_ranks = data

            with accelerator.autocast():
                pred = model(ids, mask, fts, gather_ids, sample_ids, code_pct_ranks)

            md_start=0
            code_start=0


            # pair_wise_preds=[]
            # pair_wise_targets=[]
            new_pred=[]
            for p in pred:
                target_segment=target[md_start:md_start+p.shape[0]].expand(p.shape[1],p.shape[0]).permute(1,0)
                code_pct_ranks_segment=code_pct_ranks[code_start:code_start+p.shape[1]].expand(p.shape[0],p.shape[1])
                predicted=(code_pct_ranks_segment+p).mean(1)
                new_pred.append(predicted)
                md_start+=p.shape[0]
                code_start+=p.shape[1]

            new_pred=torch.cat(new_pred)

            #print(new_pred.shape)

            assert len(target)==len(new_pred)

            pred=new_pred

            pred=accelerator.pad_across_processes(pred,pad_index=-1e9)
            pred=accelerator.gather(pred)
            # print(target)
            # print(indices)
            # exit()
            #indices=indices.cuda().long()
            indices=accelerator.pad_across_processes(indices,pad_index=-1)
            indices=accelerator.gather(indices)

            preds.append(pred.cpu())
            rearrange_indices.append(indices.cpu())
            #labels.append(target.cpu())

    preds=torch.cat(preds).numpy()
    rearrange_indices=torch.cat(rearrange_indices).numpy()
    return rearrange_indices,preds


def train(model, train_loader, val_loader, epochs, accelerator):
    np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * len(train_loader) / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    criterion = torch.nn.L1Loss(reduction='none')
    #criterion = torch.nn.MSELoss()
    #scaler = torch.cuda.amp.GradScaler()

    #accelerator = Accelerator(accelerate.DistributedDataParallelKwargs(find_unused_parameters=True))

    if args.load_weights_from != "":
        model.load_state_dict(torch.load(args.load_weights_from,map_location=accelerator.device))
    #model.load_state_dict(torch.load("../test3/outputs/model1.bin"))
    #model,optimizer,train_loader,scheduler=accelerator.prepare(model,optimizer,train_loader,scheduler)
    #model,optimizer,train_loader,scheduler=accelerator.prepare(model,optimizer,train_loader,scheduler)
    model,optimizer,train_loader,val_loader,scheduler=accelerator.prepare(model,optimizer,train_loader,val_loader,scheduler)

    for e in range(epochs):
        #if e>-1:
        train_ds = MarkdownDataset(train_df_mark.sample(frac=1,random_state=e), tokenizer=tokenizer, md_max_len=args.md_max_len,
                                   total_max_len=args.total_max_len, fts=train_fts, preds_per_forward=args.preds_per_forward)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                                  pin_memory=False, drop_last=True, collate_fn=CustomCollate(tokenizer))

        train_loader=accelerator.prepare(train_loader)

        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            if args.val_only:
                break
            #inputs, target = read_data(data)
            ids, mask, fts, gather_ids, indices, target, sample_ids, code_pct_ranks = data

            # print(code_pct_ranks)
            # exit()

            with accelerator.autocast():
                pred = model(ids, mask, fts, gather_ids, sample_ids, code_pct_ranks)


                loss=0
                md_start=0
                code_start=0


                pair_wise_preds=[]
                pair_wise_targets=[]
                for p in pred:
                    #loss+=criterion(p, target[]).mean()
                    target_segment=target[md_start:md_start+p.shape[0]].expand(p.shape[1],p.shape[0]).permute(1,0)
                    code_pct_ranks_segment=code_pct_ranks[code_start:code_start+p.shape[1]].expand(p.shape[0],p.shape[1])

                    #print(target_segment.shape)
                    #print(code_pct_ranks_segment.shape)

                    #sample_target=target[start:start+len(p)].expand(len(p),len)-code_pct_ranks[start:start+len(p)].reshape(1,-1)
                    md_start+=p.shape[0]
                    code_start+=p.shape[1]
                    #loss+=criterion(p, target_segment-code_pct_ranks_segment).mean()

                    pair_wise_preds.append(p.reshape(-1))
                    pair_wise_targets.append((target_segment-code_pct_ranks_segment).reshape(-1))
                    #print(sample_target.shape)
                    #print(p.shape)
                    #exit()
                #exit()

                pair_wise_preds=torch.cat(pair_wise_preds)
                pair_wise_targets=torch.cat(pair_wise_targets)
                loss = criterion(pair_wise_preds, pair_wise_targets).mean()#+0.5*criterion(code_pred, code_pct_ranks).mean()
                #print(loss)
                #loss = loss[:,0].mean()
            #scaler.scale(loss).backward()
            accelerator.backward(loss)
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
                #scaler.step(optimizer)
                #scaler.update()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            #preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")
            #break


        val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
        y_val, y_pred = validate(model, val_loader, accelerator)
        y_pred=y_pred[y_pred>-1e2]
        #.cpu().numpy()
        n_preds=(val_df["cell_type"] == "markdown").sum()
        y_pred=y_pred[:n_preds]
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
        #print(len(y_pred))
        # print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
        # exit()

        y_preds_tta=[y_pred]
        if args.tta>0:

            for t in range(args.tta):

                val_ds = MarkdownDataset(val_df_mark.sample(frac=1), tokenizer=tokenizer, md_max_len=args.md_max_len,
                                         total_max_len=args.total_max_len, fts=val_fts, preds_per_forward=args.preds_per_forward)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                                        pin_memory=False, drop_last=False, collate_fn=CustomCollate(tokenizer))
                val_loader=accelerator.prepare(val_loader)
                rearrange_indices, y_pred_ = validate(model, val_loader, accelerator)
                rearrange_indices=rearrange_indices.astype('int')
                #y_pred_=y_pred_[rearrange_indices!=-1]
                #print(rearrange_indices)
                #exit()
                new_preds=[]
                for i in range(n_preds):
                    index=np.where(rearrange_indices==i)[0][0]
                    #print(index)
                    new_preds.append(y_pred_[index])
                #print(new_preds[:20])
                y_preds_tta.append(np.array(new_preds))

        y_preds_tta=np.stack(y_preds_tta,0).mean(0)

        #y_pred=y_pred[:(val_df["cell_type"] == "markdown").sum()]
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_preds_tta
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)

        val_df.to_csv("val.csv",index=False)

        val_score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
        print("Preds score", val_score)
        if args.val_only:
            exit()
        #exit()
        #torch.save(model.state_dict(), f"./outputs/model{e}.bin")
        accelerator.save(accelerator.unwrap_model(model).state_dict(),f"./outputs/model{e}.bin")
        #accelerator.wait_for_everyone()
        # if e==2:

        logger.log([e,avg_loss,val_score])
    return model, y_pred


model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs, accelerator=accelerator)
