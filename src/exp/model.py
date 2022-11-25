import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModelWithLMHead


class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        #model = AutoModel.from_pretrained(model_path)
        model = AutoModelWithLMHead.from_pretrained(model_path)
        model.load_state_dict(torch.load('pytorch_model.bin'))
        self.model_path=model_path
        self.embeddings=model.deberta.embeddings
        self.encoder=model.deberta.encoder
        self.top = nn.Linear(self.embeddings.word_embeddings.embedding_dim*2+5, self.embeddings.word_embeddings.embedding_dim)
        self.gru = nn.GRU(self.embeddings.word_embeddings.embedding_dim,self.embeddings.word_embeddings.embedding_dim,bidirectional=True, dropout=0.2,batch_first=True)
        self.top2 = nn.Linear(self.embeddings.word_embeddings.embedding_dim*2, 1)
        #del self.model.pooler.dense.bias
        #del self.model.pooler.dense.weight

    def forward(self, ids, mask, fts, gather_ids, sample_ids, code_pct_ranks):
        x=self.embeddings(ids)
        # print(mask.shape)
        # print(x.shape)
        # print(mask)
        #exit()
        x=self.encoder(x,attention_mask=(mask==1),return_dict=False)[0]

        preds=[]
        code_preds=[]
        start=0
        for i in range(len(x)):

            n_code=-(gather_ids[i]).min()
            tmp=[]
            for j in range(1,n_code+1):
                vector=x[i][gather_ids[i]==-j]
                mean_vector=vector.mean(0)
                mean_vector=torch.cat((mean_vector,code_pct_ranks[start+j-1].reshape(1)))
                tmp.append(mean_vector)
            #code_preds.append(torch.stack(tmp))
            code_preds=torch.stack(tmp)

            # print(code_preds.shape)
            # exit()

            n_preds=gather_ids[i].max()
            tmp=[]
            start=0
            for j in range(1,n_preds+1):
                vector=x[i][gather_ids[i]==j]
                # if return_vectors:
                #     vectors.append(vector)
                mean_vector=vector.mean(0)
                mean_vector=torch.cat((mean_vector, fts[i]))

                mean_vector=mean_vector.expand(len(code_preds),len(mean_vector)) #n_codexC
                # print(code_preds.shape)
                # print(mean_vector.shape)
                # exit()


                mean_vector=torch.cat([mean_vector,code_preds],1)
                tmp.append(mean_vector)

            tmp=torch.stack(tmp)
            merged=F.relu(self.top(tmp))
            merged=self.gru(merged)[0]
            #max_merged=merged.mean(1)

            # print(max_merged.shape)
            # print(n_preds)
            # exit()
            preds.append(self.top2(merged).squeeze(-1))



            start+=len(code_preds)





            #preds.append(torch.stack(tmp))
        #preds=torch.cat(preds)

        # print(preds.shape)
        # exit()

        #preds = self.top2(preds)
        #code_preds = self.top2(torch.cat(code_preds))
        # mean_preds=[]
        # for i in range(torch.max(sample_ids)+1):
        #     indices=torch.where(sample_ids==i)[0]
        #     tmp=[]
        #     for j in indices:
        #         tmp.append(preds[j])
        #     tmp=torch.stack(tmp).max(0)[0]
        #     mean_preds.append(tmp)
        #
        #
        # mean_preds=torch.cat(mean_preds)
        # preds = self.top(mean_preds)


        return preds
