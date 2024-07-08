import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import os
from SummarizationDataset import SummarizationDataset
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from train_model import train_model
import wandb
import math

wandb.init(project="bart-large-runs", config={
    "learning_rate": 5e-5,
    "architecture": "t5",
    "dataset": "Discharge Me!",
    "epochs": 3,
    })

os.environ['HF_HOME'] = '/vol/csedu-nobackup/project/lnguyen/.cache'


RADIOLOGY_REPORT_SEP = "<RAD_SEP>"
ICD_TITLE_TOKEN = "</s>"
CHIEF_COMPLAINT_TOKEN = "</s>"
RADIOLOGY_TOKEN = "</s>"

df_aggregated_train = pd.read_csv('dataframeagg_train_entities.csv', keep_default_na=False)
df_aggregated_val = pd.read_csv('dataframeagg_val_entities.csv', keep_default_na=False)

model_name = 'facebook/bart-large'
tokenizer = BartTokenizer.from_pretrained(model_name)
model_course = BartForConditionalGeneration.from_pretrained(model_name)

if ICD_TITLE_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([ICD_TITLE_TOKEN])
if CHIEF_COMPLAINT_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([CHIEF_COMPLAINT_TOKEN])
if RADIOLOGY_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_tokens([RADIOLOGY_TOKEN])
if RADIOLOGY_REPORT_SEP not in tokenizer.get_vocab():
    tokenizer.add_tokens([RADIOLOGY_REPORT_SEP])


texts_train = df_aggregated_train['preprocessed_discharge_text'].tolist()
# texts_train = ["summarize: " + text for text in texts_train]

summaries_train_course = df_aggregated_train['brief_hospital_course'].tolist()
texts_val = df_aggregated_val['preprocessed_discharge_text'].tolist()
# texts_val = ["summarize: " + text for text in texts_val]

summaries_val_course = df_aggregated_val['brief_hospital_course'].tolist()

dataset_train_course = SummarizationDataset(texts_train, summaries_train_course, tokenizer, max_input_length=512, max_target_length=512)
loader_train_course = DataLoader(dataset_train_course, batch_size=4, shuffle=True)

dataset_val_course = SummarizationDataset(texts_val, summaries_val_course, tokenizer, max_input_length=512, max_target_length=512)
loader_val_course = DataLoader(dataset_val_course, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_course.to(device)
model_save_path = './model_checkpoints'
epochs = 3

optimizer_course = AdamW(model_course.parameters(), lr=5e-5)
total_steps_course = len(loader_train_course) * epochs
scheduler_course = get_linear_schedule_with_warmup(optimizer_course, num_warmup_steps=0, num_training_steps=total_steps_course)


if __name__ == '__main__':
    train_model(model_course, loader_train_course, loader_val_course, optimizer_course, 'course', 'bart-large', scheduler_course)