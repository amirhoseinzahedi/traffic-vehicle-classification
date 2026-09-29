# CNN Baseline

## 1. Objective

The purpose of this experiment is to establish a simple CNN baseline for the traffic-vehicle classification task.

The baseline provides a reference point for later controlled experiments involving augmentation, regularization, optimization, class balancing, loss functions, scheduling, and transfer learning.

The test set is kept frozen and is not used during baseline training or model selection.

---

## 2. Dataset

The cleaned training dataset contains 400 images from 8 vehicle classes.

A reproducible stratified split with seed `42` was used:

| Split      | Samples |
| ---------- | ------: |
| Training   |     320 |
| Validation |      80 |
| Test       |     400 |

Each class contributes:

* 40 training samples
* 10 validation samples
* 50 test samples

The validation indices are stored in:

```text
configs/validation_indices.npy
```

The test set was not used for model selection.

---

## 3. Class Mapping

The class mapping used by the model is:

| Index | Class     |
| ----: | --------- |
|     0 | ambulance |
|     1 | autobus   |
|     2 | kamyun    |
|     3 | kamyunet  |
|     4 | minibus   |
|     5 | savari    |
|     6 | taxi      |
|     7 | vanet     |

The same mapping is stored in the model checkpoint.

---

## 4. Model Architecture

The baseline model is a small CNN named `BaselineCNN`.

It contains two convolutional blocks:

```text
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool
```

The resulting feature representation is passed to a fully connected classifier:

```text
Flatten

→ Linear

→ ReLU

→ Linear
```

The final layer produces 8 logits, one for each vehicle class.

Input image size:

```text
3 × 128 × 128
```

The model contains approximately 8.4 million trainable parameters.

The large majority of the parameters are in the first fully connected layer.

---

## 5. Training Configuration

| Configuration     | Value                |
| ----------------- | -------------------- |
| Device            | CUDA                 |
| Image size        | 128 × 128            |
| Batch size        | 32                   |
| Epochs            | 10                   |
| Optimizer         | Adam                 |
| Learning rate     | 0.001                |
| Loss function     | CrossEntropyLoss     |
| Random seed       | 42                   |
| Data augmentation | RandomHorizontalFlip |
| Scheduler         | None                 |
| Weight decay      | None                 |
| Dropout           | None                 |

The learning rate remained constant during this experiment.

---

## 6. Training Results

The training history was recorded in:

```text
reports/baseline_history.csv
```

The results were:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ----: | ---------: | --------: | -------: | ------: |
|     1 |     2.1455 |    0.1562 |   2.0236 |  0.3000 |
|     2 |     1.9844 |    0.2844 |   1.9246 |  0.3625 |
|     3 |     1.7957 |    0.3969 |   1.6501 |  0.4750 |
|     4 |     1.5069 |    0.4781 |   1.6617 |  0.3375 |
|     5 |     1.2951 |    0.5625 |   1.4233 |  0.5125 |
|     6 |     1.0563 |    0.6750 |   1.1987 |  0.6000 |
|     7 |     0.8130 |    0.7406 |   1.1337 |  0.5500 |
|     8 |     0.6391 |    0.7906 |   1.0248 |  0.6125 |
|     9 |     0.4980 |    0.8344 |   1.0492 |  0.6500 |
|    10 |     0.3866 |    0.8781 |   1.0403 |  0.6750 |

Best validation accuracy:

```text
67.50%
```

The best checkpoint corresponds to:

```text
Epoch: 10
Validation Accuracy: 67.50%
```

Validation accuracy improved from 30.00% at epoch 1 to 67.50% at epoch 10.

The best validation accuracy was therefore achieved at the final training epoch.

---

## 7. Learning Curves

The following plots were generated:

```text
reports/figures/baseline_loss_curve.png

reports/figures/baseline_accuracy_curve.png

reports/figures/baseline_learning_rate.png
```

### Loss

Training loss decreased consistently throughout the experiment, from `2.1455` at epoch 1 to `0.3866` at epoch 10.

Validation loss generally decreased during training, reaching its minimum of approximately `1.0248` at epoch 8. It then increased slightly at epoch 9 and remained close to that level at epoch 10.

The divergence between training and validation loss in the later epochs suggests that the model continues to fit the training data more strongly than the validation data.

### Accuracy

Training accuracy increased from `15.62%` to `87.81%`.

Validation accuracy increased from `30.00%` to a maximum of `67.50%`.

The largest validation accuracy was obtained at epoch 10, while training accuracy continued to increase throughout the experiment.

By epoch 10, the gap between training and validation accuracy was:

```text
87.81% - 67.50% = 20.31 percentage points
```

This indicates a substantial generalization gap and motivates later experiments involving regularization and other controlled changes.

### Learning Rate

The learning rate remained constant throughout the experiment:

```text
0.001
```

No learning-rate scheduler was used in the baseline.

---

## 8. Checkpoint

The best model is stored at:

```text
checkpoints/baseline_cnn_best.pth
```

The checkpoint was verified successfully.

Stored metadata includes:

```text
architecture

num_classes

class_mapping

image_size

batch_size

optimizer

learning_rate

loss_function

seed

epoch

validation_accuracy
```

Checkpoint verification result:

```text
PASS
```

The stored checkpoint corresponds to epoch 10 and has a validation accuracy of 67.50%.

---

## 9. Interpretation

The baseline demonstrates that the CNN can learn meaningful classification patterns from the training data.

Training accuracy increased to `87.81%`, while the best validation accuracy reached `67.50%`.

The difference between training and validation accuracy at epoch 10 is approximately `20.31` percentage points, indicating a substantial generalization gap.

Validation loss reached its lowest value at epoch 8 (`1.0248`), while validation accuracy continued to improve through epoch 10.

This means that validation loss and validation accuracy do not identify exactly the same epoch as the best training checkpoint criterion. Since checkpoint selection is based on validation accuracy, epoch 10 is the selected baseline checkpoint.

The baseline establishes a reference point for the controlled experiments that follow.

No conclusion about final test-set performance is made at this stage because the test set has intentionally remained untouched.

---

## 10. Reproducibility Artifacts

The following artifacts are used to reproduce and inspect the baseline experiment:

```text
src/dataset.py

src/model.py

src/train.py

src/plot_training.py

src/verify_checkpoint.py

configs/validation_indices.npy

reports/baseline_history.csv

reports/figures/baseline_loss_curve.png

reports/figures/baseline_accuracy_curve.png

reports/figures/baseline_learning_rate.png

checkpoints/baseline_cnn_best.pth
```

The dataset itself is confidential and is not included in the repository.
