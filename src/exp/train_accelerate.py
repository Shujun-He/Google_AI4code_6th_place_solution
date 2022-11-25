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


parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str, default='microsoft/deberta-base')
parser.add_argument('--train_mark_path', type=str, default='./../data/train_mark.csv')
parser.add_argument('--train_features_path', type=str, default='./../data/train_fts.json')
parser.add_argument('--val_mark_path', type=str, default='./../data/val_mark.csv')
parser.add_argument('--val_features_path', type=str, default='./../data/val_fts.json')
parser.add_argument('--val_path', type=str, default="./../data/val.csv")

parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=3e-5)

torch.multiprocessing.set_sharing_strategy('file_system')

args = parser.parse_args()
os.system("mkdir ./outputs")
data_dir = Path('../../input/')
os.environ["TORCH_DISTRIBUTED_DEBUG"]="INFO" #0,1,2,3 for four gpu

train_df_mark = pd.read_csv(args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
train_fts = json.load(open(args.train_features_path))
val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
val_fts = json.load(open(args.val_features_path))
val_df = pd.read_csv(args.val_path)

order_df = pd.read_csv("../../input/train_orders.csv").set_index("id")
df_orders = pd.read_csv(
    data_dir / 'train_orders.csv',
    index_col='id',
    squeeze=True,
).str.split()

#exit()

train_ds = MarkdownDataset(train_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                           total_max_len=args.total_max_len, fts=train_fts)

def read_data(batch):
    ids, mask, fts, gather_ids, labels =[],[],[],[],[]
    # print(len(batch))
    # exit()
    for data in batch:
        ids.append(data[0])
        mask.append(data[1])
        fts.append(data[2])
        gather_ids.append(data[3])
        labels.append(data[4])
        #print(data[4])
    #exit()

    #print(labels)
    #exit()

    # ids, mask, fts, gather_ids=tuple(d.cuda() for d in data[:-1])
    # torch.cat(data[-1]).cuda()
    #ids=torch.stack(ids)#.cuda()
    #mask=torch.stack(mask)#.cuda()
    #fts=torch.stack(fts)#.cuda()
    #gather_ids=torch.stack(gather_ids)#.cuda()
    #labels=torch.cat(labels)#.cuda()

    ids=torch.stack(ids)#.cuda()
    mask=torch.stack(mask)#.cuda()
    fts=torch.stack(fts)#.cuda()
    gather_ids=torch.stack(gather_ids)#.cuda()
    labels=torch.cat(labels)#.cuda()


    return (ids, mask, fts, gather_ids), labels

#train_ds[0]
val_ds = MarkdownDataset(val_df_mark, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len, fts=val_fts)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                          pin_memory=False, drop_last=True, collate_fn=read_data)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False, collate_fn=read_data)





def validate(model, val_loader, accelerator):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):

        #for idx, data in enumerate(val_loader):
            inputs, target = data
            #print(idx)
            #print(target.shape)

            with torch.cuda.amp.autocast():
            #with accelerator.autocast():
                pred = model(*inputs)

            #preds.append(pred.detach().cpu().numpy().ravel())
            #labels.append(target.detach().cpu().numpy().ravel())
                #pred=pred#.cpu()
                #if accelerator.is_main_process:
            # if idx!=(len(val_loader)-1):
            #     pred=accelerator.gather(pred)
            #     target=accelerator.gather(target)
            # else:
            #     #pred=pred[:len(val_ds)%(torch.distributed.get_world_size()*args.batch_size)]
            #     pass
            #pred=accelerator.gather(pred)
            #target=accelerator.gather(target)

            preds.append(pred.cpu())
            labels.append(target.cpu())

    labels,preds=torch.cat(labels), torch.cat(preds)
    # print(len(val_ds))
    # print(labels.shape)
    # print(preds.shape)
    # exit()
    return labels,preds


def train(model, train_loader, val_loader, epochs):
    np.random.seed(0)
    # Creating optimizer and lr schedulers
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
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

    accelerator = Accelerator(accelerate.DistributedDataParallelKwargs(find_unused_parameters=True))

    #model,optimizer,train_loader,scheduler=accelerator.prepare(model,optimizer,train_loader,scheduler)
    model,optimizer,train_loader,scheduler=accelerator.prepare(model,optimizer,train_loader,scheduler)


    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            #inputs, target = read_data(data)
            inputs, target = data
            #ids, mask, fts=inputs
            # print(target.shape)
            # exit()
            #with torch.cuda.amp.autocast():
            with accelerator.autocast():
                pred = model(*inputs)

                # print(pred.shape)
                # print(target.shape)
                # exit()
                #target=target.expand(512,args.batch_size,1).permute(1,0,2)
                #exit()



                loss = criterion(pred, target).mean()
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
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")
            #break

        #if accelerator.is_main_process:
        y_val, y_pred = validate(model, val_loader, accelerator)
        y_pred=y_pred.cpu().numpy()
        #y_pred = accelerator.gather(y_pred).cpu().numpy()
        #if accelerator.is_main_process:

        val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

        y_pred=y_pred[:(val_df["cell_type"] == "markdown").sum()]
        val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
        print("Preds score", kendall_tau(df_orders.loc[y_dummy.index], y_dummy))
        #torch.save(model.state_dict(), f"./outputs/model{e}.bin")
        accelerator.save(accelerator.unwrap_model(model).state_dict(),f"./outputs/model{e}.bin")
        #accelerator.wait_for_everyone()
        # if e==2:
        #     exit()
    return model, y_pred


model = MarkdownModel(args.model_name_or_path)
model = model.cuda()
model, y_pred = train(model, train_loader, val_loader, epochs=args.epochs)
