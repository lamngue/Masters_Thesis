import pandas as pd
import os
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
os.chdir('./test_phase_2')

discharge_target = pd.read_csv('discharge_target.csv.gz', keep_default_na=False)
discharge = pd.read_csv('discharge.csv.gz', keep_default_na=False)
discharge.rename({"text": "discharge_text"}, axis='columns', inplace =True)

radiology = pd.read_csv('radiology.csv.gz', keep_default_na=False)
radiology.rename({"text": "radiology_text"}, axis='columns', inplace =True)

diagnosis = pd.read_csv('diagnosis.csv.gz', keep_default_na=False)
edstays = pd.read_csv('edstays.csv.gz', keep_default_na=False)
triage = pd.read_csv('triage.csv.gz', keep_default_na=False)
triage = triage[['stay_id', 'subject_id', 'chiefcomplaint']]


def categorize_by_word_count(row):
    if row['discharge_instructions_word_count'] < 100:
        return 'short'
    elif row['discharge_instructions_word_count'] <= 300:
        return 'medium'
    else:
        return 'long'

discharge_target['instruction_length'] = discharge_target.apply(categorize_by_word_count, axis=1)

discharge_target = pd.merge(discharge_target, discharge, on="hadm_id")
discharge_target = discharge_target.drop(['discharge_instructions_word_count'], axis=1)

discharge_enriched = pd.merge(edstays, discharge_target, on="hadm_id")
discharge_diagnosis_enriched = pd.merge(discharge_enriched, diagnosis, on="stay_id")

discharge_diagnosis_enriched = discharge_diagnosis_enriched.drop(['subject_id_x'], axis=1)

discharge_diagnosis_radiology = pd.merge(discharge_diagnosis_enriched, radiology, on="hadm_id")
discharge_diagnosis_radiology.drop(discharge_diagnosis_radiology.columns[[3,4,5,6,7]], axis=1, inplace=True)
discharge_diagnosis_radiology_triage = pd.merge(discharge_diagnosis_radiology, triage, on="stay_id")
discharge_diagnosis_radiology_triage = discharge_diagnosis_radiology_triage.drop(['note_type_y', 'note_seq_y', 'charttime_y','storetime_y', 'icd_code', 'icd_version'], axis=1)

del discharge_target
del discharge
del radiology
del triage
del diagnosis
del edstays

def mask_text(discharge_text, discharge_instructions, hospital_course):
    # Replace the target sections with a placeholder or remove them
    if discharge_instructions in discharge_text:
        discharge_text = discharge_text.replace(discharge_instructions, ' ')
    if hospital_course in discharge_text:
        discharge_text = discharge_text.replace(hospital_course, ' ')
    return discharge_text


discharge_diagnosis_radiology_triage['masked_discharge_text'] = discharge_diagnosis_radiology_triage.apply(
    lambda x: mask_text(x['discharge_text'], x['discharge_instructions'], x['brief_hospital_course']), axis=1
)


df_cleaned = discharge_diagnosis_radiology_triage.drop(['note_id_x', 'subject_id_x', 'note_id_y', 'subject_id_y','note_seq_x', 'note_type_x', 'charttime_x', 'storetime_x'], axis=1)

aggregated_icd_titles = df_cleaned.groupby('hadm_id')['icd_title'].apply(set).reset_index()

aggregated_chiefcomplaint = df_cleaned.groupby('hadm_id')['chiefcomplaint'].apply(set).reset_index()

aggregated_icd_titles.rename(columns={'icd_title': 'aggregated_icd_titles'}, inplace=True)

aggregated_chiefcomplaint.rename(columns={'chiefcomplaint': 'aggregated_chiefcomplaint'}, inplace=True)

stop_words = set(stopwords.words('english'))
custom_punctuation = string.punctuation.replace(".", "")

def simplify_sentence(sentence):
    tokens = word_tokenize(sentence)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in custom_punctuation]
    return " ".join(filtered_tokens)

def simplify_radiology_texts(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    simplified_sentences = [simplify_sentence(sentence) for sentence in sentences]
    return " ".join(simplified_sentences)


def extract_keywords(text, num_keywords=20):
    if not text.strip():
        return []  # Return an empty list if the text is empty or only whitespace
    try:
        vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
        return feature_array[tfidf_sorting].tolist()
    except ValueError as e:
        print(f"Error in extract_keywords: {e}")
        return []
    
nlp = spacy.load("en_ner_bc5cdr_md")  # Example of a medical NER model


def extract_medical_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    unique_entities = list(set(entities))  # Remove duplicates by converting to a set and back to a list
    return unique_entities

def rank_sentences(text, reference_sentences, vectorizer, tfidf_matrix):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Preprocess each sentence
    # preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

    # Transform the sentences using the vectorizer
    sentences_matrix = vectorizer.transform(sentences)

    # Create a TF-IDF matrix for the reference themes
    reference_matrix = vectorizer.transform(reference_sentences)

    # Calculate cosine similarity between the sentences and reference themes
    similarities = cosine_similarity(sentences_matrix, reference_matrix)

    # Sum the similarity scores across all themes for each sentence
    relevance_scores = similarities.sum(axis=1)

    # Zip scores with sentences and sort them by relevance score
    scored_sentences = sorted(zip(relevance_scores, sentences), reverse=True, key=lambda x: x[0])

    # Extract top sentences (you can adjust the number as per your requirement)
    top_sentences = [sentence for score, sentence in scored_sentences[:20]]  # Top 20 sentences

    return ' '.join(top_sentences)

RADIOLOGY_REPORT_SEP = "<RAD_SEP>"
aggregated_radiology_texts = df_cleaned.groupby('hadm_id')['radiology_text'].apply(lambda texts: RADIOLOGY_REPORT_SEP.join(texts)).reset_index()
aggregated_radiology_texts.rename(columns={'radiology_text': 'joined_radiology_text'}, inplace=True)

df_cleaned = pd.merge(df_cleaned, aggregated_icd_titles, on='hadm_id', how='left')
df_cleaned = pd.merge(df_cleaned, aggregated_chiefcomplaint, on='hadm_id', how='left')
df_cleaned = pd.merge(df_cleaned, aggregated_radiology_texts, on='hadm_id', how='left')

ICD_TITLE_TOKEN = "</s>"
CHIEF_COMPLAINT_TOKEN = "</s>"
RADIOLOGY_TOKEN = "</s>"

df_cleaned['aggregated_icd_titles_str'] = df_cleaned['aggregated_icd_titles'].apply(lambda titles: "; ".join(titles))
df_cleaned['aggregated_chiefcomplaint'] = df_cleaned['aggregated_chiefcomplaint'].apply(lambda complaint: ''.join(complaint))
df_cleaned['aggregated_radiologies'] = df_cleaned['radiology_text']

df_cleaned['masked_discharge_text'] = df_cleaned['masked_discharge_text'] + ICD_TITLE_TOKEN + df_cleaned['aggregated_icd_titles_str']
df_cleaned['masked_discharge_text'] = df_cleaned['masked_discharge_text'] + CHIEF_COMPLAINT_TOKEN + df_cleaned['aggregated_chiefcomplaint']
df_cleaned['masked_discharge_text'] = df_cleaned['masked_discharge_text'] + RADIOLOGY_TOKEN + "Radiologies-Section: " + df_cleaned['aggregated_radiologies']

def remove_intro_text(text):
    # The regex pattern is designed to match the specified intro text more flexibly
    pattern = re.compile(
        r'^\s*Name:\s*___\s*Unit\s*No:\s*___\s*'
        r'Admission\s*Date:\s*___\s*Discharge\s*Date:\s*___\s*'
        r'Date\s*of\s*Birth:\s*___\s*Sex:\s*[MF]\s*', re.IGNORECASE
    )
    return re.sub(pattern, '', text)


def simplify_discharge_texts(text):
    sentences = sent_tokenize(text)
    simplified_sentences = [simplify_sentence(sentence) for sentence in sentences]
    return " ".join(simplified_sentences)


def simplify_radiology_section(text):
    # Split the radiology section using `<RAD_SEP>`
    individual_reports = re.split(r'<RAD_SEP>', text)
    # Simplify each report individually
    simplified_reports = [simplify_radiology_texts(report) for report in individual_reports]
    return "<RAD_SEP>".join(simplified_reports)

def keywords_discharge_texts(text):
    # Extract top keywords
    keywords = list(set(extract_keywords(text)))
    # Filter out keywords that are already in the simplified sentences
    final_text = ", ".join(keywords)
    return final_text

def keywords_radiology_section(text):
    # Split the radiology section using `<RAD_SEP>`
    individual_reports = re.split(r'<RAD_SEP>', text)
    # entities each report individually
    entities_reports = [keywords_discharge_texts(report) for report in individual_reports]

    return "<RAD_SEP>".join(entities_reports)

def entities_discharge_texts(text):
    # Extract top entities
    entities = extract_medical_entities(text)
    # Filter out entities that are already in the simplified sentences
    final_text = ", ".join([f"{ent[0]}" for ent in entities])
    return final_text

def entities_radiology_section(text):
    # Split the radiology section using `<RAD_SEP>`
    individual_reports = re.split(r'<RAD_SEP>', text)
    # entities each report individually
    entities_reports = [entities_discharge_texts(report) for report in individual_reports]

    return "<RAD_SEP>".join(entities_reports)

def simplify_sections(text):
    # Split by the general section separator `</s>`
    sections = re.split(r'</s>', text)
    simplified_sections = []

    for section in sections:
        # Check if the section is the radiology section and simplify using `<RAD_SEP>`
        if section.startswith("Radiologies-Section:"):
            simplified_section = simplify_radiology_section(section)
        else:
            simplified_section = simplify_discharge_texts(section)
        simplified_sections.append(simplified_section)

    return "</s>".join(simplified_sections)

def keywords_sections(text):
    # Split by the general section separator `</s>`
    sections = re.split(r'</s>', text)
    keywords_sections = []
    keywords = ""
    for i, section in enumerate(sections):
        if i == 0:
            keywords += keywords_discharge_texts(section) + " "
        elif i == 3:
            keywords += keywords_radiology_section(section) + " "
        keywords_sections.append(section)

    return "Keywords: " + keywords + "</s> Discharge text: " + " ".join(keywords_sections)

def entities_sections(text):
    # Split by the general section separator `</s>`
    sections = re.split(r'</s>', text)
    entities_sections = []
    entities = ""
    for i, section in enumerate(sections):
        if i == 0:
            entities += entities_discharge_texts(section) + " "
        elif i == 3:
            entities += entities_radiology_section(section) + " "
        entities_sections.append(section)

    return "Entities: " + entities + "</s> Discharge text: " + " ".join(entities_sections)


def preprocess_text(discharge_text):
    # Placeholder removal, standardizing newlines, lowercasing, and removing header information
    discharge_text = re.sub(r'___', '', discharge_text)
    discharge_text = re.sub(r'\s+\.', '.', discharge_text)
    discharge_text = re.sub(r'\n\s*\n', '\n', discharge_text).strip().lower()
    discharge_text = re.sub(r'^\nname: .*?\n', '', discharge_text, flags=re.DOTALL)

    # Remove special characters sequences like =, *, and similar
    discharge_text = re.sub(r'[=]{2,}', ' ', discharge_text)  # Replace sequences of '=' with a space
    discharge_text = re.sub(r'[*]{2,}', ' ', discharge_text)  # Replace sequences of '*' with a space
    discharge_text = re.sub(r'[-]{2,}', ' ', discharge_text)  # Replace sequences of '-' with a space
    discharge_text = re.sub(r'[~]{2,}', ' ', discharge_text)  # Replace sequences of '~' with a space

    # Remove newlines and extra spaces
    discharge_text = re.sub(r'\n', ' ', discharge_text)
    discharge_text = re.sub(r'\s+', ' ', discharge_text).strip()

    return discharge_text

df_aggregated = df_cleaned.groupby('hadm_id').agg({
    'masked_discharge_text': 'first',  # Assuming the text is the same for each 'hadm_id'
    'brief_hospital_course': 'first',
    'discharge_instructions': 'first'
}).reset_index()

# reference_sentences = [
#     "discharge summary",
#     "patient diagnosis",
#     "treatment plan",
#     "follow-up instructions",
#     "radiology findings"
# ]
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(df_aggregated['masked_discharge_text'])

# df_aggregated['ranked_masked_discharge_text'] = df_aggregated['masked_discharge_text'].apply(
#     lambda x: rank_sentences(x, reference_sentences, vectorizer, tfidf_matrix)
# )

df_aggregated['trimmed_discharge_text'] = df_aggregated['masked_discharge_text'].apply(remove_intro_text)

df_aggregated['entities_discharge_text'] = df_aggregated['trimmed_discharge_text'].apply(entities_sections)
df_aggregated['preprocessed_discharge_text'] = df_aggregated['entities_discharge_text'].apply(preprocess_text)

path = '/vol/csedu-nobackup/project/lnguyen/Thesis/dataframeagg_test_phase_2_entities.csv'
with open(path, 'w', encoding = 'utf-8-sig') as f:
  df_aggregated.to_csv(f)

def split_sentences(ranked_sentences):
    hospital_course = []
    discharge_instructions = []
    is_discharge_instruction = False
    is_hospital_course = False

    for sentence in ranked_sentences:
        sentence = sentence.replace("</s>", "").replace("<RAD_SEP>", "").strip()
        if not sentence:
            continue  # Skip empty sentences

        # Check for keywords indicating a switch in sections
        sentence_lower = sentence.lower().strip()

        if sentence_lower.startswith(('discharge instructions', 'follow-up', 'follow up')) or any(kw in sentence.lower() for kw in ['follow-up', 'medication', 'instructions']):
            is_discharge_instruction = True
            is_hospital_course = False
        elif sentence_lower.startswith(('hospital course', 'brief hospital course')) or any(kw in sentence.lower() for kw in ['hospital course', 'admission', 'treatment', 'complaint']):
            is_hospital_course = True
            is_discharge_instruction = False

        # Append sentences to the respective lists
        if is_discharge_instruction:
            discharge_instructions.append(sentence)
        elif is_hospital_course:
            hospital_course.append(sentence)
     # Ensure there are no empty inputs
    if not hospital_course:
        hospital_course.append("No hospital course provided.")
    if not discharge_instructions:
        discharge_instructions.append("No discharge instructions provided.")

    return ' '.join(hospital_course), ' '.join(discharge_instructions)


def ensure_non_empty_columns(df, col1, col2, default1="No hospital course provided.", default2="No discharge instructions provided."):
    df[col1] = df[col1].apply(lambda x: x if x.strip() else default1)
    df[col2] = df[col2].apply(lambda x: x if x.strip() else default2)
    return df

df_aggregated['ranked_masked_discharge_text_list'] = df_aggregated['preprocessed_discharge_text'].apply(lambda x: re.split(r'(?<=[.!?])\s+|\n', x))
baseline_df = pd.DataFrame()
baseline_df['hadm_id'] = df_aggregated['hadm_id']

baseline_df[['brief_hospital_course', 'discharge_instructions']] = df_aggregated['ranked_masked_discharge_text_list'].apply(
    lambda x: pd.Series(split_sentences(x))
)

baseline_df = ensure_non_empty_columns(baseline_df, 'brief_hospital_course', 'discharge_instructions')

path = '/vol/csedu-nobackup/project/lnguyen/Thesis/baseline_submissions.csv'
with open(path, 'w', encoding = 'utf-8-sig') as f:
  baseline_df.to_csv(f)
