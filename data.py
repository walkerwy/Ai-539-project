import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import coco_dataset_ratio, device
from preprocessing import preprocess, collate_fn

def load_coco_datasets():
    train_ds = load_dataset("HuggingFaceM4/COCO", split=f"train[:{coco_dataset_ratio}%]")
    valid_ds = load_dataset("HuggingFaceM4/COCO", split=f"validation[:{coco_dataset_ratio}%]")
    test_ds = load_dataset("HuggingFaceM4/COCO", split="test")

    train_ds = train_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=4)
    valid_ds = valid_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=4)
    test_ds = test_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=4)

    return train_ds, valid_ds, test_ds

def get_transformed_datasets():
    train_ds, valid_ds, test_ds = load_coco_datasets()
    train_dataset = train_ds.with_transform(preprocess)
    valid_dataset = valid_ds.with_transform(preprocess)
    test_dataset = test_ds.with_transform(preprocess)
    return train_dataset, valid_dataset, test_dataset

def get_data_loaders(batch_size):
    train_dataset, valid_dataset, test_dataset = get_transformed_datasets()
    train_dataset_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    valid_dataset_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    test_dataset_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_dataset_loader
