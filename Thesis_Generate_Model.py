import os
import pandas as pd

from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer

from SummarizationDataset import SummarizationDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = './model_checkpoints'
tokenizer_instructions = AutoTokenizer.from_pretrained('luqh/ClinicalT5-base')

RADIOLOGY_REPORT_SEP = "<RAD_SEP>"
ICD_TITLE_TOKEN = "</s>"
CHIEF_COMPLAINT_TOKEN = "</s>"
RADIOLOGY_TOKEN = "</s>"

if ICD_TITLE_TOKEN not in tokenizer_instructions.get_vocab():
    tokenizer_instructions.add_tokens([ICD_TITLE_TOKEN])
if CHIEF_COMPLAINT_TOKEN not in tokenizer_instructions.get_vocab():
    tokenizer_instructions.add_tokens([CHIEF_COMPLAINT_TOKEN])
if RADIOLOGY_TOKEN not in tokenizer_instructions.get_vocab():
    tokenizer_instructions.add_tokens([RADIOLOGY_TOKEN])
if RADIOLOGY_REPORT_SEP not in tokenizer_instructions.get_vocab():
    tokenizer_instructions.add_tokens([RADIOLOGY_REPORT_SEP])

df_aggregated_test_phase_2 = pd.read_csv('dataframeagg_test_phase_2_entities.csv', keep_default_na=False)
texts_test_phase_2 = df_aggregated_test_phase_2['preprocessed_discharge_text'].tolist()
summaries_test_phase_2_instructions = df_aggregated_test_phase_2['discharge_instructions'].tolist()

trained_model_instructions = T5ForConditionalGeneration.from_pretrained('luqh/ClinicalT5-base', gradient_checkpointing=True, from_flax=True).to(device)
model_instructions_path = os.path.join(model_save_path, 'model_entities_epoch_3_instructions_clinical-t5.pt')
trained_model_instructions.load_state_dict(torch.load(model_instructions_path))

trained_model_instructions.to(device)  # Make sure to move your model to the appropriate device (GPU or CPU)
trained_model_instructions.eval()

dataset_test_phase_2_instructions = SummarizationDataset(
    texts=texts_test_phase_2,
    summaries=summaries_test_phase_2_instructions,
    tokenizer=tokenizer_instructions,
    max_input_length=512,
    max_target_length=512
)

test_phase_2_loader_instructions = DataLoader(
    dataset_test_phase_2_instructions,
    batch_size=16,  # Or another batch size that fits your system
    shuffle=False,  # Typically, you don't need to shuffle test data
    pin_memory=True,  # Enable fast data transfer to GPU
    num_workers=2
)

if __name__ == '__main__':
    import pickle
    predictions_instructions = []

    with torch.no_grad():
        total_batches = len(test_phase_2_loader_instructions)

        for batch in tqdm(test_phase_2_loader_instructions, desc="Generating Predictions Instructions", total=total_batches, unit="batch"):
            assert batch['input_ids'].size(1) <= 512
            with autocast():  # Mixed precision
                outputs = trained_model_instructions.generate(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    max_length=550,
                    num_beams=5,
                    early_stopping=True
                )
            decoded_preds = tokenizer_instructions.batch_decode(outputs, skip_special_tokens=True)
            predictions_instructions.extend(decoded_preds)
    with open("predictions_instructions_clinical-t5_entities", "wb") as fp:
        pickle.dump(predictions_instructions, fp)