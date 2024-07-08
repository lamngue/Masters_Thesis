from transformers import BertTokenizer, BertModel
import spacy
import numpy as np
import pandas as pd
import torch
import pickle
import random

random.seed(42)

# Load ClinicalBERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
bert_model = BertModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# Load a medical NER model
nlp = spacy.load("en_ner_bc5cdr_md")

def calculate_semantic_similarity(text1, text2):
    # Tokenize and encode the texts
    inputs1 = bert_tokenizer(text1, return_tensors='pt', truncation=True, max_length=512)
    inputs2 = bert_tokenizer(text2, return_tensors='pt', truncation=True, max_length=512)
    
    # Get the embeddings
    with torch.no_grad():
        embeddings1 = bert_model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = bert_model(**inputs2).last_hidden_state.mean(dim=1)
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    return similarity.item()

def extract_medical_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def evaluate_with_ai(input_text, generated_summary):
    # Calculate semantic similarity
    semantic_similarity = calculate_semantic_similarity(input_text, generated_summary)
    
    # Extract entities
    input_entities = extract_medical_entities(input_text)
    generated_entities = extract_medical_entities(generated_summary)
    
     # Calculate entity overlap
    if len(input_entities) == 0:
        entity_overlap = 0  # Handle case where there are no input entities
    else:
        common_entities = set(input_entities) & set(generated_entities)
        entity_overlap = len(common_entities) / len(set(input_entities))
    
    return semantic_similarity, entity_overlap

# Load predictions
with open("predictions_course_mBart_base", "rb") as fp:
    predictions_course = pickle.load(fp)

with open("predictions_instructions_mBart_base", "rb") as fp:
    predictions_instructions = pickle.load(fp)

discharge_target_text = pd.read_csv('discharge_target_reference_mBart.csv')

# Ensure both predictions have the same length
assert len(predictions_course) == len(predictions_instructions)

# Randomly select 20 indices
indices = random.sample(range(len(predictions_course)), 3)

# Sample 20 rows randomly from the DataFrame based on the selected indices
sampled_discharge_target_text = discharge_target_text.iloc[indices].reset_index(drop=True)

# Select the predictions based on the sampled indices
sampled_predictions_course = [predictions_course[i] for i in indices]
sampled_predictions_instructions = [predictions_instructions[i] for i in indices]

# Initialize lists to store the similarities and entity overlaps
generated_course_arr = []
reference_course_arr = []

generated_instructions_arr = []
reference_instructions_arr = []

# course_entity_overlaps = []
instructions_similarities = []
# instructions_entity_overlaps = []

# Assuming 'sampled_predictions_course' & 'sampled_predictions_instructions' are your predictions
for idx, row in sampled_discharge_target_text.iterrows():
    input_course = row['brief_hospital_course']
    input_instructions = row['discharge_instructions']
    
    generated_course = sampled_predictions_course[idx]
    generated_instructions = sampled_predictions_instructions[idx]
    
    # course_similarity, course_entity_overlap = evaluate_with_ai(input_course, generated_course)
    # instruction_similarity, instruction_entity_overlap = evaluate_with_ai(input_instructions, generated_instructions)
    reference_course_arr.append(input_course)
    generated_course_arr.append(generated_course)
    reference_instructions_arr.append(input_instructions)
    generated_instructions_arr.append(generated_instructions)
    # course_similarities.append(course_similarity)
    # course_entity_overlaps.append(course_entity_overlap)
    # instructions_similarities.append(instruction_similarity)
    # instructions_entity_overlaps.append(instruction_entity_overlap)

res_course = "\n".join("{} {}".format(x, y) for x, y in zip(generated_course_arr, reference_course_arr))
res_instructions = "\n".join("{} {}".format(x, y) for x, y in zip(generated_instructions_arr, reference_instructions_arr))
# Create DataFrames for summary statistics
# course_similarity_stats = pd.DataFrame(course_similarities, columns=["similarity"]).describe()
# course_entity_overlap_stats = pd.DataFrame(course_entity_overlaps, columns=["entity_overlap"]).describe()
# instructions_similarity_stats = pd.DataFrame(instructions_similarities, columns=["similarity"]).describe()
# instructions_entity_overlap_stats = pd.DataFrame(instructions_entity_overlaps, columns=["entity_overlap"]).describe()
print("GENERATED TEXT")
print(generated_instructions_arr)
print("REFERENCE TEXT")
print(reference_instructions_arr)
# # Calculate summary statistics
# course_similarity_stats = pd.DataFrame(course_similarities, columns=["similarity"]).describe()
# course_entity_overlap_stats = pd.DataFrame(course_entity_overlaps, columns=["entity_overlap"]).describe()
# instructions_similarity_stats = pd.DataFrame(instructions_similarities, columns=["similarity"]).describe()
# instructions_entity_overlap_stats = pd.DataFrame(instructions_entity_overlaps, columns=["entity_overlap"]).describe()

# print('STATISTICS FOR HOSPITAL COURSE SIMILARITY: \n', course_similarity_stats)
# print('STATISTICS FOR HOSPITAL COURSE ENTITY OVERLAP: \n', course_entity_overlap_stats)
# print('STATISTICS FOR DISCHARGE INSTRUCTIONS SIMILARITY: \n', instructions_similarity_stats)
# print('STATISTICS FOR DISCHARGE INSTRUCTIONS ENTITY OVERLAP: \n',instructions_entity_overlap_stats)