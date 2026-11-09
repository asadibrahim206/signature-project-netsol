# Signature Verification - Model Training

## Model
Contrastive Siamese Convolutional Neural Network (CNN)

## Dataset
Handwritten Signature Dataset:
- Genuine signatures
- Forged signatures
- 600 signature pairs

## Model Architecture
- Input: 128 × 128 grayscale images
- Shared Siamese CNN encoder
- Embedding size: 128
- Total parameters: 110,080

## Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 64% |
| ROC-AUC | 0.74 |
| EER | 36% |
| FAR | 36% |
| FRR | 36% |
| ECE | 13.09% |

## Training Artifacts
- Model architecture
- Training and validation loss graph
- ROC curve
- Confusion matrix
- Classification report
- FAR and FRR results
- ECE result
- Practical signature test

Note: Evaluation was performed on a reconstructed owner-wise test split of 100 pairs.
