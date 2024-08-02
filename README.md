# Detection of Cigarette Cabinet Openings Using CCTV: A Supermarket Case Study

## Authors
Daniela Pereira Rigoli, Luigi Carvalho and Lucas Silveira Kupssinskü 

## Project Overview

This project aims to detect the opening of cigarette cabinets in a supermarket using CCTV footage. The detection is performed using a deep learning model implemented with PyTorch and Keras, and it involves preprocessing video frames, training a model, and evaluating its performance.

## Repository Contents

- `main.py`: This script contains the implementation, testing, and validation of the VGG model used for cabinet opening detection.
- `ProcessAux.py`: This script handles the preprocessing of video frames, which are categorized into 'open' and 'close' subfolders.
- `generateCutFrames.ipynb`: A Jupyter Notebook for preprocessing video frames in segments.
- `getByLog.py`: A script for extracting logs from the cashier terminal to correlate with CCTV footage.

## Requirements

To run this project, ensure you have the following Python packages installed:

- numpy
- torch
- torchvision
- scikit-learn
- matplotlib
- opencv-python
- keras

## Installation

You can install the required packages using pip. Run the following command:

```bash
pip install numpy torch torchvision scikit-learn matplotlib opencv-python keras
```

## Usage
1. Extract Logs:

Use getByLog.py to extract logs from the cashier terminal for additional analysis or correlation with detected events. This need be adapted for other projects. And if you already have the videos you don't need worry about it.

2. Preprocess Video Frames:

Use ProcessAux.py to preprocess video frames into 'open' and 'close' categories. You can also use generateCutFrames.ipynb for preprocessing in parts.

3. Train and Validate the Model:

Run main.py to train the VGG model on the preprocessed data. This script will also handle testing and validation.

## Model and Methods
The project employs a deep learning approach using a VGG model architecture implemented in PyTorch. The key steps include:

* Data Preprocessing: Video frames are extracted and labeled as 'open' or 'close.'

* Model Training: The VGG model is trained on the preprocessed dataset.

* Evaluation: The model is evaluated using metrics like accuracy, precision, recall, F1-score, ROC curve, and AUC.

# Results
The results of the model are evaluated using various performance metrics to ensure accuracy and reliability in detecting cabinet openings.

