from torch.utils.data import Dataset, DataLoader
import torch

from torch.utils.data import Dataset, DataLoader
import torch

class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_input_length=512, max_target_length=512):
        self.texts = texts
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_text = "summarize: " + self.texts[idx]
        tokenized_input = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        tokenized_target = self.tokenizer(
            self.summaries[idx],
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized_input['input_ids'].squeeze(0),
            'attention_mask': tokenized_input['attention_mask'].squeeze(0),
            'labels': tokenized_target['input_ids'].squeeze(0)
        }