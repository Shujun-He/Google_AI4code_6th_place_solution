import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig


model_name="microsoft/deberta-v3-base"
if True:
    os.system('mkdir model')

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.save_pretrained('model')

    config_model = AutoConfig.from_pretrained(model_name)
    config_model.num_labels = 15
    config_model.save_pretrained('model')

    backbone = AutoModelForTokenClassification.from_pretrained(model_name,
                                                               config=config_model)
    backbone.save_pretrained('model')
