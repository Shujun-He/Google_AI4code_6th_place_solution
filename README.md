# Google_AI4code_6th_place_solution

## How to run

### Download data

`bash download_data.sh`

This will download all the data you need, provided you have Kaggle API set up already

### Preprocess data
`cd src/exp`\
`python preprocess.py`

### Run MLM pretraining
`bash pretrain.sh`

### Finetune pretrained model

first stage:\
change n_code in preprocess.py to 20\
`python preprocess.py`\
`bash tta.sh`


second stage:\
change n_code in preprocess.py to 40\
`python preprocess.py`\
`bash tta_2nd.sh`

third stage:\
change n_code in preprocess.py to 80\
`python preprocess.py`\
`bash tta_3rd.sh`

## Solution details

### Relative to absolute position prediction


![alt text](https://github.com/Shujun-He/Google_AI4code_6th_place_solution/blob/main/graphics/relative2absolute.png)

(I typed this in latex and just screenshot it. Sorry for the quality)

### Network achitecture
My network architecture consists mainly of a deberta encoder and a GRU layer; an overview is
shown in the Figure below. First, markdown blocks and code blocks are tokenized and concatenated.
During training, 10 markdown blocks with 40 code blocks with a max length of 1600 are used
first and then the weights are retrained with 20 markdown blocks with 80 code blocks with a
max length of 3160. Following that, a deberta large encoder is used to encode the text, and
then each markdown and code block is pooled. Next, features are added to pooled feature
vectors (discussed in a later section). Each markdown block feature vector is then concatenated
to the sequence of code block feature vectors, and then an GRU network is used to predict the
relative position of that markdown block to each code blocks. Lastly, the absolute position of
the markdown cell is computed according to Equation 1. MAE loss is used during training.

![alt text](https://github.com/Shujun-He/Google_AI4code_6th_place_solution/blob/main/graphics/architecture.png)






