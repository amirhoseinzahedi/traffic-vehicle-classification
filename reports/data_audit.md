# Dataset Audit

## 1. Purpose

This report documents the initial audit of the traffic-vehicle image classification dataset.

The audit was performed before creating the train/validation split or training any model. The purpose was to verify:

* class distribution
* file integrity
* image formats
* image dimensions
* representative samples
* duplicate images
* potential train/test leakage
* label conflicts
* data-quality exclusions

The original dataset is confidential and is not included in this repository.

---

## 2. Dataset Structure

The local dataset contains three splits:

```text
dataset/
├── train/
├── test/
└── unclean/
```

The expected class structure is:

* `train`: 8 known vehicle classes
* `test`: the same 8 known vehicle classes
* `unclean`: the 8 known classes plus an additional `neysan` class

The `unclean` split is kept separate from training and is intended for data-quality and uncertainty analysis.

---

## 3. Class Distribution

### Train

| Class     |  Images |
| --------- | ------: |
| ambulance |      50 |
| autobus   |      50 |
| kamyun    |      50 |
| kamyunet  |      50 |
| minibus   |      50 |
| savari    |      50 |
| taxi      |      50 |
| vanet     |      50 |
| **Total** | **400** |

The training data is balanced, with 50 images per class.

### Test

| Class     |  Images |
| --------- | ------: |
| ambulance |      50 |
| autobus   |      50 |
| kamyun    |      50 |
| kamyunet  |      50 |
| minibus   |      50 |
| savari    |      50 |
| taxi      |      50 |
| vanet     |      50 |
| **Total** | **400** |

The test data is also balanced, with 50 images per class.

### Unclean

| Class     |  Images |
| --------- | ------: |
| ambulance |      50 |
| autobus   |      50 |
| kamyun    |      50 |
| kamyunet  |      50 |
| minibus   |      50 |
| neysan    |      50 |
| savari    |      50 |
| taxi      |      50 |
| vanet     |      50 |
| **Total** | **450** |

The `unclean` split contains 50 images for each of the 9 observed classes.

---

## 4. File Integrity

Each image was opened using Pillow to verify that it could be decoded successfully.

A minimum image-size threshold of `32 × 32` pixels was used for the audit.

### Results

| Split   | Total | Invalid | Too Small | Format |
| ------- | ----: | ------: | --------: | ------ |
| train   |   400 |       0 |         0 | JPEG   |
| test    |   400 |       0 |         0 | JPEG   |
| unclean |   450 |       0 |         0 | JPEG   |

All inspected files were readable JPEG images.

No unreadable files were found.

No image was below the selected minimum-size threshold.

---

## 5. Image Dimensions

The images have varying dimensions.

### Train

* Width: 120–545 px
* Mean width: 255.8 px
* Median width: 257.0 px
* Height: 156–870 px
* Mean height: 315.4 px
* Median height: 273.5 px

### Test

* Width: 126–508 px
* Mean width: 254.2 px
* Median width: 242.5 px
* Height: 166–881 px
* Mean height: 313.1 px
* Median height: 272.0 px

### Unclean

* Width: 120–478 px
* Mean width: 255.2 px
* Median width: 254.0 px
* Height: 168–938 px
* Mean height: 316.9 px
* Median height: 277.0 px

The images therefore do not have a uniform spatial resolution.

This is not treated as a data-quality failure. Image resizing will be handled later by the model preprocessing pipeline.

---

## 6. Representative Sample Inspection

Representative images were displayed from the training set with their corresponding class labels.

The inspected samples showed:

* a visible vehicle in each displayed image
* labels that appeared compatible with the displayed vehicle
* no obvious corruption
* no obviously mislabeled sample among the displayed examples

This inspection is only a qualitative sample check. It does not establish that every image in the dataset is correctly labelled.

---

## 7. Duplicate Detection

Duplicate detection was performed using an image-content hash rather than filenames.

For each image, the decoded RGB pixel data and image dimensions were hashed using SHA-256.

This allows identical decoded image content to be detected even when filename information is not sufficient.

### Duplicate Summary

| Category        | Duplicate Groups |
| --------------- | ---------------: |
| train ↔ test    |                0 |
| train ↔ unclean |               15 |
| test ↔ unclean  |                8 |
| label conflict  |                1 |
| other           |                0 |
| **Total**       |           **24** |

---

## 8. Train/Test Leakage Analysis

No exact duplicate image was found between the training and test sets.

```text
train ↔ test = 0
```

Therefore, the audit found no direct train/test duplication based on the image-content hashing method used.

This does not rule out every possible form of dataset similarity or near-duplicate image. The implemented check detects identical decoded pixel content, not visually similar images with different crops, resizing, or other pixel-level changes.

---

## 9. Train/Unclean Duplicates

15 duplicate groups were found between `train` and `unclean`.

These images have identical content and the same observed class label.

Because the corresponding images have already appeared in the training set, they should not be treated as independent samples when evaluating the behavior of the model on `unclean`.

The original files are not deleted.

---

## 10. Test/Unclean Duplicates

8 duplicate groups were found between `test` and `unclean`.

These images are already present in the frozen test set.

Since `unclean` is not used for training, these duplicates do not constitute direct train/test leakage. However, they should not be treated as independent evidence when analyzing the `unclean` dataset.

The original files are not deleted.

---

## 11. Label Conflict

One exact duplicate was found with different labels:

```text
train/vanet/214844236.jpg
unclean/neysan/214844236.jpg
```

The two files have identical image content but different class labels:

```text
vanet ≠ neysan
```

This is treated as a label conflict.

In particular, this image cannot be considered a genuine unseen `neysan` example because the identical image content is already present in the training data under the `vanet` class.

Therefore, this sample will be excluded from evidence used to evaluate generalization to genuinely unseen `neysan` images.

The original files are retained.

---

## 12. Exclusion Policy

The audit does not modify or delete the original dataset.

The following policy will be used in later phases:

1. The `train` split remains the source for the training and validation subsets.
2. The `test` split remains frozen and is reserved for final evaluation.
3. The `unclean` split is not used for model training.
4. Exact `train ↔ unclean` duplicates are not treated as independent `unclean` samples.
5. Exact `test ↔ unclean` duplicates are not treated as independent `unclean` samples.
6. The identified `vanet ↔ neysan` label-conflict image is excluded from genuine unseen-class analysis.
7. Original dataset files are retained; exclusions are handled by analysis policy rather than deletion.
8. All future exclusions should be reproducible and documented.

---

## 13. Audit Conclusion

The audit found no unreadable images, no images below the selected minimum-size threshold, and no direct train/test duplicate images.

The training and test sets are both class-balanced.

The main data-quality findings concern the `unclean` split:

* 15 exact duplicates with training data
* 8 exact duplicates with test data
* 1 exact duplicate with conflicting labels

The identified train/test duplication result supports proceeding to the next phase without an observed direct train/test leakage issue.

The `unclean` dataset will remain isolated and will be analyzed separately after model development, with duplicate and label-conflict cases handled according to the policy documented above.

---

## 14. Reproducibility

The audit was implemented in:

```text
src/data_audit.py
```

The dataset itself is confidential and excluded from version control.

The audit script and this report are version-controlled so that the methodology and conclusions can be reviewed independently of the private dataset.
