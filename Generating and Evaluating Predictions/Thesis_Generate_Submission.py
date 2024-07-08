from transformers import BartTokenizer, AutoTokenizer, T5ForConditionalGeneration
import pandas as pd
import numpy as np
import os
import pickle
import torch 

os.chdir('./')

discharge_target = pd.read_csv('baseline_submissions.csv', keep_default_na=False)

brief_hospital_courses = discharge_target['brief_hospital_course'].tolist()
discharge_instructions = discharge_target['discharge_instructions'].tolist()

tokenizer_course = BartTokenizer.from_pretrained('facebook/bart-large')
tokenizer_instructions = BartTokenizer.from_pretrained('facebook/bart-large')

special_tokens = ["</s>", "<RAD_SEP>"]
tokenizer_course.add_tokens(special_tokens, special_tokens=True)
tokenizer_instructions.add_tokens(special_tokens, special_tokens=True)

def safe_tokenize(tokenizer, texts, max_length=512):
    non_empty_texts = [text for text in texts if text.strip()]  # Filter out empty texts
    if not non_empty_texts:
        return {'input_ids': torch.tensor(['empty']), 'attention_mask': torch.tensor([['empty']])}
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # Ensure no empty tensors
    input_ids = [ids for ids in tokenized['input_ids'] if ids.numel() > 0]
    attention_mask = [mask for mask in tokenized['attention_mask'] if mask.numel() > 0]
    if not input_ids or not attention_mask:
        return {'input_ids': torch.tensor(['empty']), 'attention_mask': torch.tensor(['empty'])}
    return {'input_ids': torch.stack(input_ids), 'attention_mask': torch.stack(attention_mask)}

tokenized_courses = safe_tokenize(tokenizer_course, brief_hospital_courses, max_length=512)
tokenized_instructions = safe_tokenize(tokenizer_instructions, discharge_instructions, max_length=512)

decoded_courses = [tokenizer_course.decode(ids, skip_special_tokens=True) for ids in tokenized_courses['input_ids']]
decoded_instructions = [tokenizer_instructions.decode(ids, skip_special_tokens=True) for ids in tokenized_instructions['input_ids']]

# Create a new dataframe for saving
discharge_target_text = pd.DataFrame({
    'hadm_id': discharge_target['hadm_id'],
    'brief_hospital_course': decoded_courses,
    'discharge_instructions': decoded_instructions
})

discharge_target_text.to_csv('baseline_submission_mBart.csv', index=False)
# df_aggregated_test_phase_2 = pd.read_csv('dataframeagg_test_phase_2.csv', keep_default_na=False)
# with open("predictions_course_clinical-t5_entities", "rb") as fp:   # Unpickling
#    predictions_course = pickle.load(fp)

# with open("predictions_instructions_clinical-t5_entities", "rb") as fp:   # Unpickling
#    predictions_instructions = pickle.load(fp)


# discharge_target_text = pd.read_csv('discharge_target_reference_clinical-t5.csv', keep_default_na=False)
# hadm_ids = [id for id in df_aggregated_test_phase_2['hadm_id']]
# submission_df = pd.DataFrame({
#     'hadm_id': hadm_ids,
#     'brief_hospital_course': predictions_course,
#     'discharge_instructions': predictions_instructions
# })
# submission_df = submission_df.set_index('hadm_id').reindex(discharge_target_text['hadm_id']).reset_index()
# submission_df.to_csv('submissions_entities_clinical-t5.csv', index=False, quoting=1)