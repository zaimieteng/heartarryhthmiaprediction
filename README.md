# ECG Signal Classification — Convolutional Neural Network

Deep learning project to classify electrocardiogram (ECG) signals into
5 cardiac rhythm classes using a 1D Convolutional Neural Network (CNN),
aimed at supporting early detection of cardiovascular diseases.

## Problem Statement
Traditional ECG analysis is time-consuming, requires professional
cardiologists, and is prone to subjective variability and misdiagnosis.
This project proposes a CNN model to automatically classify ECG signals
and reduce reliance on manual interpretation.

## Key Findings
- Final tuned model achieved 83.62% accuracy, up from 82.53% baseline
- Bootstrap resampling balanced the dataset from ~80% Normal class
  dominance to 20,000 samples per class
- Gaussian noise augmentation (mean=0, std=0.5) improved model
  generalisation to real-world signal variations
- Best hyperparameters via GridSearchCV: batch size 32, 40 epochs,
  dropout 0.1, RMSProp optimiser

## Model Results

| Model | Accuracy |
|---|---|
| Initial Model | 82.53% |
| Final Model (tuned) | 83.62% |

## Heart Rhythm Classes

| Class | Label | Description |
|---|---|---|
| Normal | N | Normal heart rhythm |
| Supraventricular | S | Extra beats in the supraventricular region |
| Ventricular | V | Extra beats in the ventricular region |
| Fusion | F | More than one source of pacemaker impulses |
| Unclassifiable | Q | Signals that cannot be classified |

## Network Architecture
Input → Conv1D (×3 blocks) → Batch Normalisation → Max Pooling
→ Flatten → Dense + Dropout → Softmax (5 classes)

- 3 convolutional blocks with kernel sizes 6 and 3, ReLU activation
- Batch normalisation for stable training
- Dropout layers to prevent overfitting
- Early stopping and model checkpointing during training
- Learning rate scheduling applied

## Methodology
1. Loaded PhysioNet MIT-BIH dataset — 109,446 ECG beats across 5 classes
2. Applied bootstrap undersampling to balance classes to 20,000 each
3. Added Gaussian noise for data augmentation and generalisation
4. Built CNN architecture with 3 convolutional blocks
5. Tuned hyperparameters using GridSearchCV with 3-fold cross validation
6. Evaluated on 20% held-out test set (21,892 beats)

## Tools Used
Python — TensorFlow, Keras, NumPy, scikit-learn, matplotlib

## Data Source
PhysioNet MIT-BIH Arrhythmia Database — ECG recordings from 47
individuals, sampled at 360 Hz, annotated by professional cardiologists.
[Dataset Link](https://www.physionet.org/content/mitdb/1.0.0/)
