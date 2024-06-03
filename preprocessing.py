import torch
from transformers import ViTImageProcessor, GPT2TokenizerFast
from config import device, max_length, encoder_model, decoder_model

tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
image_processor = ViTImageProcessor.from_pretrained(encoder_model)

def preprocess(items):
    pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
    targets = tokenizer([sentence["raw"] for sentence in items["sentences"]],
                        max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    return {'pixel_values': pixel_values, 'labels': targets["input_ids"]}

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }
