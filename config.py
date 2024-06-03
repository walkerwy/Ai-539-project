import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 32
coco_dataset_ratio = 50
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
patience = 3
weight_decay = 1e-5
encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
decoder_model = "gpt2"
