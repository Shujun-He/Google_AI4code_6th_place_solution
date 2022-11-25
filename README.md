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
