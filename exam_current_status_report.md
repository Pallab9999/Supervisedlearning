# Current Exam Project Status Report

## Project Context

This project addresses the Machine Learning for Modelling supervised learning exam task: food image classification on the iFood-2019 dataset using a custom CNN for supervised learning (SL), then comparing it with a CNN trained using self-supervised learning (SSL) and evaluated through a traditional classifier on extracted features.

The current notebook is `app.ipynb`. The experiment is running on Windows with CUDA enabled:

- PyTorch: `2.6.0+cu124`
- Device: `cuda`
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU

## Dataset Setup

The dataset is correctly available and loaded from:

`ifood-2019-fgvc6`

Current dataset facts:

| Item | Current Status |
|---|---:|
| Training images available | 118,475 |
| Validation/test images available | 11,994 |
| Classes | 251 |
| Dataset percentage used | 10% |
| Internal SL training split | 9,484 images |
| Internal SL validation split | 2,371 images |
| Final test subset from official validation set | 1,207 images |

This satisfies the exam constraint that all 251 classes must be retained, while allowing a reduced number of images per class for computational reasons.

## Supervised Learning Pipeline

The supervised model is a custom CNN inspired by VGG-16, with four convolutional blocks, batch normalization, dropout, adaptive average pooling, and a fully connected classifier.

Current model size:

| Model | Parameters | Constraint |
|---|---:|---|
| Custom CNN `Net` | 7,915,931 | Under 10M |

This satisfies the exam requirement that the CNN must be custom and have fewer than 10 million parameters.

The supervised training pipeline includes:

- Image resizing to `160x160`
- RGB image loading
- Data augmentation: rotation, horizontal flip, Gaussian blur, random resized crop
- Normalization
- Weighted cross-entropy loss
- AdamW optimizer
- Cosine annealing learning-rate scheduler
- Best-model checkpoint saving as `best_model_weights.pth`

### SL Training Status

The original SL model was stuck near random performance. A later SL run improved the training curves:

- Training loss decreased from about `5.53` to about `5.14`
- Validation loss decreased at first and stabilized around `5.40`
- Training accuracy increased to about `3%`
- Validation accuracy increased to about `2.1%`

This shows the model began learning after the training transform was corrected and training was extended.

### SL Test Result

The latest pasted SL test result is:

| Metric | SL Result |
|---|---:|
| Test accuracy | 0.0033 |
| Macro F1 | 0.0000 |
| Weighted F1 | 0.0000 |

The classification report shows that the model mostly predicts one class, producing almost zero precision/recall/F1 across most classes. This is still not a strong supervised result.

## Self-Supervised Learning Pipeline

The SSL pipeline implements SimCLR:

- Paired augmented views of each image
- CNN encoder based on the same `Net` convolutional backbone
- Projection head with output dimension 128
- NT-Xent contrastive loss
- AdamW optimizer
- Cosine annealing learning-rate scheduler
- Best SSL checkpoint saved as `best_model_ssl.pth`

The SSL model uses all sampled training images without labels:

| SSL Item | Value |
|---|---:|
| SSL images | 11,855 |
| SSL batches | 185 |
| SSL epochs | 10 |
| SSL training time | 508.33 minutes |
| SSL parameters | 7,804,864 |

### SSL Training Result

The SSL contrastive loss decreased steadily:

| Epoch | NT-Xent Loss |
|---:|---:|
| 1 | 4.8445 |
| 2 | 4.8442 |
| 3 | 4.5879 |
| 4 | 4.4127 |
| 5 | 4.2028 |
| 6 | 4.0520 |
| 7 | 3.9150 |
| 8 | 3.8257 |
| 9 | 3.7944 |
| 10 | 3.7767 |

This indicates that the contrastive model learned a representation from the unlabeled images.

## SSL Feature Extraction and Traditional Classifier

After SSL training, encoder features were extracted:

| Feature Split | Shape |
|---|---:|
| Training features | `(9484, 512)` |
| Test features | `(1207, 512)` |

A logistic regression classifier was trained on these frozen SSL features.

Current SSL linear-probe results:

| Metric | SSL + Logistic Regression |
|---|---:|
| Accuracy | 0.0307 |
| Macro F1 | 0.0243 |
| Weighted F1 | 0.0259 |

SSL currently performs better than SL in the final classification comparison.

## Current SL vs SSL Comparison

| Metric | Supervised Learning | Self-Supervised Learning |
|---|---:|---:|
| Accuracy | 0.0033 latest pasted / 0.0050 stale saved output | 0.0307 |
| Macro F1 | 0.0000 | 0.0243 |
| Weighted F1 | 0.0000 | 0.0259 |
| Parameters | 7,915,931 | 7,804,864 |
| Training time | about 278 minutes saved output | 508.33 minutes |
| Approach | End-to-end supervised CNN | SimCLR + logistic regression |

The saved notebook comparison table still contains stale SL values from an older Cell 26 run. It should be rerun after the final SL evaluation if those values are intended to appear in the notebook.

## Exam Requirement Coverage

| Exam Requirement | Current Status |
|---|---|
| Data preprocessing: resize, RGB, normalize | Implemented |
| Custom CNN | Implemented |
| CNN under 10M parameters | Satisfied |
| All 251 classes retained | Satisfied |
| Dataset reduction allowed for computational reasons | Used 10% |
| Validation set extracted from training set | Implemented |
| SL formulation | Implemented |
| SSL formulation | Implemented |
| Feature extraction from SSL model | Implemented |
| Traditional classifier on SSL features | Implemented |
| Accuracy, precision, recall, F1 | Implemented |
| Visualizations | Mostly implemented |
| Hyperparameter tuning | Partially implemented |
| Technical report | Not yet finalized |
| Presentation | Not yet prepared |

## Main Limitations to Discuss With Peer

1. The SL model performs very poorly on the test set despite improved training curves.

2. The current SL classification report suggests class collapse, where the model predicts mainly one class.

3. A likely technical issue is the normalization scale. The notebook uses mean/std values in the `0-255` pixel scale, but `transforms.ToTensor()` converts images to `0-1`. Correct normalization would divide mean and std by 255. Fixing this would be more correct, but it would affect both SL and SSL. Since SSL took about 8 hours, this needs discussion before changing.

4. The project uses only 10% of the dataset. This is acceptable for computational reasons, but it limits performance.

5. The custom CNN is trained from scratch with no pretrained weights, making the 251-class task difficult.

6. SSL was trained for only 10 epochs. SimCLR usually benefits from many more epochs and larger batches.

7. The final report must be careful not to overclaim performance. The correct story is that the pipeline is complete, SSL currently outperforms SL, but absolute performance remains low.

## Recommended Next Decisions

### Option A: Keep Current Results

Use the current results as an honest experimental comparison:

- Advantage: no more 8-hour SSL training.
- Disadvantage: SL result is weak and must be explained as a limitation.

This is acceptable if the report clearly discusses the normalization issue, limited data, custom model size, and low training budget.

### Option B: Fix Normalization and Retrain Only SL

Change mean/std to `0-1` scale and retrain SL only:

- Advantage: SL may improve significantly.
- Disadvantage: comparison is less fair because SSL was trained with the old normalization.

This can be presented as an additional SL tuning experiment, but the report must say SSL was not retrained.

### Option C: Fix Normalization and Retrain Both SL and SSL

This is the most methodologically correct option:

- Advantage: fair and cleaner comparison.
- Disadvantage: SSL retraining could take another 8 hours.

## Suggested Report Framing

The final report should present the work as a complete implementation of both paradigms, with a transparent discussion of computational limitations:

- The custom CNN satisfies the exam constraints.
- The SSL pipeline is fully implemented and produces useful features.
- The SSL linear probe outperforms the SL baseline in the current run.
- Absolute performance is low due to data reduction, no pretrained weights, limited epochs, small batch size for SimCLR, and possible normalization scaling.
- Future work should include full-dataset training, corrected normalization, deeper hyperparameter tuning, and optionally fine-tuning the SSL encoder.

## Required Report Statement

The report should include a plagiarism statement, for example:

> I declare that this project report and code were prepared without plagiarism. Any external sources, documentation, libraries, or assistance used for understanding or implementation are acknowledged appropriately.

