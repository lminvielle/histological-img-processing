# Histopathology image analysis
This repository contains a short project over histopatholoy images, with two
tasks: classification of images containing (or not) nuclei and a nuclei localization
algorithm.

## Setup
This code runs on Python >= 3.8. If possible, use conda and set up an
environment with:

- `conda create -n hist_img_analysis python=3.8`

- `conda activate hist_img_analysis`

- `pip install -r requirements.txt`

## Usage

- For the default paths to work, the data folder (containing _classification_ and _localization_ image folders) should be placed next to this repository, otherwise paths have to be changed.
- Scripts to execute models are respectively located in code/classification and
  code/localization.

### Classification

In **code/classification** folder, running `python classification.py` will successively train a classification model, test it, display
scores and record output classified images.

All model parameters can be set in the *Params* section of the script.

### Localization

In **code/localization** folder, running `python localization.py` will successively prepare data, train Yolo and test it over
the test set.
These three steps can be activated or deactivated in the *Params* section.

- Data preparation

Before any training can be done, label files must be created in accordance with Yolo
training.

- Training

During training, Yolo automatically creates a _run_ folder in which it will place
all results and weights.

**Avoiding training**:
Since the training can take quite some time, a weight file (named _best.pt_) corresponding to Yolo
size L trained over RGB images is available [here](https://drive.google.com/file/d/1usxGy7x4s5XRdzY1pvKDUlSZ-sOSok-O/view?usp=sharing).
Place it in **runs/train/sizeL\_clrRGB/weights**
The command line to create the whole path is `mkdir -p runs/train/sizeL_clrRGB/weights` (to be run from folder **localization**).
In this case, you can skip data preparation and training and directly execute
the testing part of the script.

- Testing

This part of the script will load a trained Yolo model, launch it over the testing
dataset and save predictions (as images).
