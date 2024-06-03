import argparse
from torchvision import transforms
import os
import numpy as np

from process import CocoCaptionDataset
# from model import PoseEstimationModel
from train import train_model
from model import 
import matplotlib.pyplot as plt

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--images", required=True, type=str,
                                        help="Specify path to images directory.")
    parser.add_argument('-a', "--anno", required=True, type=str,
                                        help="Specify path to annotations file (csv).")
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()

    images_path = "/nfs/stak/users/arulmozg/hpc-share/Gen-Alt-txt/coco/images"
    annotations_path = "/nfs/stak/users/arulmozg/hpc-share/Gen-Alt-txt/coco/annotations/annotations/captions_train2017.json"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    coco_dataset = CocoCaptionDataset(
        root_dir= images_path,
        ann_file= annotations_path
        transform=transform
    )

    dataloader = DataLoader(coco_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Further processing for the Swin Transformer and encoder-decoder model

    
    print("Training Phase")

    # model = StackedHourGlass(nChannels=256, nStack=2, nModules=2, numReductions=4, nJoints=num_keypoints)
    train_model(model, dataset, batch_size=64, num_epochs=10, learning_rate=1e-3, device='cuda')

