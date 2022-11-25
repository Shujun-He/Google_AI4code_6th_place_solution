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

(I typed this in latex and just screenshot it. Sorry for the quality. For a better version, see `write_up_6th_solution.pdf`)

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

### Features

The following features are used:

1. number of markdown cells in given notebook (normalized by 128)\
2. number of code cells in given notebook (normalized by 128)\
3. ratio of markdown cells to total number of cells\

These features are concatenated with pooled feature vectors of markdown cells. Pooled code
cells feature vectors are also concatenated with their absolute positions.

### MLM
Masked langauge modeling pretraining is used, where the input text is arranged in the same
manner as in finetuning. MLM probability was set to 0.15 and the model is trained for 10
epochs. A max sequence length of 768 is used with at most 5 markdown cells and 10 code cells.

### Inference and TTA
During inference, I tried to predict positions of as many markdown cells as possible. I was
able to use a max length of 4250 with 100 code blocks and 30 markdown blocks. Code blocks
have at most 26 tokens and markdown blocks 64. Notebooks are sorted based on number of
markdown/cell blocks to reduce inference time. Because for longer sequences where there are
more than 30 markdown blocks, I cannot make predictions for all of them at the same time,
and multiple forward passes are needed to make predictions for all markdown blocks. As a
result, the model never sees all the markdown blocks at the same time. To counteract that, I
do test time augmentation by randomly shuffling the markdown cells each time a new inference
cycle is started and therefore the model will see different combinations of markdown cells in
long sequences. For shorter sequences, this simply changes the order of markdown cells during
inference. For my best inference notebook, a total of 3 inference cycles are used (3xTTA). 3xTTA improves from score from 0.9067 to 0.9100.

### What didn’t work
1. Trying to translate markdown cells that are not in English did not result in any improve-
ments for me. I tried facebook’s m2m100 418M.\
2. Separating code and markdown cells and doing two forward passes through deberta for
concatenated code and markdown cells.\
3. Cubert did not work because I couldn’t fix the tokenizer.\
4. Maxpooling instead of meanpooling

### References
I used the solution from Khoi Nguyen as a baseline: https://www.kaggle.com/competitions/
AI4Code/discussion/326970.


