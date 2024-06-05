# ImageCaptionGen
Please change the README.md if new files are added or you want to write more instructions.
I did my best, but some of the information here or in the requirements.txt may be incorrect.

## File System Strcture
```
ImageCaptionGen
|   /gitignore                  - Ignores data in /coco
|   coco.sh                     - Downloads coco datasets
|   config.py                   - Holds hyper parameters
|   data.py                     - Creates data loaders
|   fine-tuned-model.ipynb      - I'm not sure
|   LICENSE                     - The project's license
|   model.py                    - The model definition
|   preprocessing.py            - Tokenizes and pads text
|   README.md                   - The readme file
|   requirements.txt            - List of Python packages
|   run.py                      - Creates and trains model
|   train.py                    - Training loop and evaluation
|   utils.py                    - Computes test metrics
|
|_______coco                    - Cache for COCO dataset
|
|_______models
        |   model_epoch_X.pt    - Best val accuracy save
        |   model_at_last.pt    - Model at last epoc
        |
        |_______checkpoint      - Holds training checkpoints
```

## Using this project
1. Install everything from requirements.txt.
2. Run the "full-loop.ipynb" file.