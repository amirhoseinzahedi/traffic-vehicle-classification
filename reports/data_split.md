# Data Split Report

## 1. Objective

This phase establishes the training, validation, and test data protocol for the traffic-vehicle classification project.

The main objectives are:

* Create a reproducible stratified validation split.
* Preserve the class distribution between training and validation subsets.
* Verify that the training and test datasets use the same class mapping.
* Keep the test set completely separate for final evaluation.
* Keep the `unclean` dataset separate from model training and validation.

---

## 2. Dataset Configuration

The cleaned training dataset contains 400 images across 8 vehicle classes.

The dataset is divided into the following logical subsets:

| Dataset   | Purpose                               | Images |
| --------- | ------------------------------------- | -----: |
| `train`   | Source for training and validation    |    400 |
| `test`    | Final evaluation                      |    400 |
| `unclean` | Data-quality and uncertainty analysis |    450 |

The `test` dataset is not used during model development or model selection.

The `unclean` dataset is not used to create the training or validation subsets.

---

## 3. Class Mapping

The training and test datasets use the following identical class-to-index mapping:

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

The mapping was verified programmatically before creating the validation split.

---

## 4. Validation Split

A stratified train/validation split was created from the cleaned `train` dataset.

### Configuration

| Parameter          | Value           |
| ------------------ | --------------- |
| Source dataset     | `dataset/train` |
| Total samples      | 400             |
| Validation ratio   | 20%             |
| Training samples   | 320             |
| Validation samples | 80              |
| Random seed        | 42              |
| Split method       | Stratified      |

Stratification ensures that each class maintains the same proportion in the training and validation subsets.

---

## 5. Resulting Class Distribution

The resulting split contains exactly 40 training samples and 10 validation samples for every class.

| Class     | Training | Validation |   Total |
| --------- | -------: | ---------: | ------: |
| ambulance |       40 |         10 |      50 |
| autobus   |       40 |         10 |      50 |
| kamyun    |       40 |         10 |      50 |
| kamyunet  |       40 |         10 |      50 |
| minibus   |       40 |         10 |      50 |
| savari    |       40 |         10 |      50 |
| taxi      |       40 |         10 |      50 |
| vanet     |       40 |         10 |      50 |
| **Total** |  **320** |     **80** | **400** |

No class is over- or under-represented in the resulting validation split.

---

## 6. Reproducibility

The validation indices are generated using a fixed random seed:

```text
SEED = 42
```

The selected validation indices are saved to:

```text
configs/validation_indices.npy
```

The split was regenerated using the same seed and configuration. The resulting validation indices were compared with the original indices.

Result:

```text
Validation indices are reproducible: PASS
```

This ensures that subsequent experiments can use the same validation subset rather than generating a different split for each experiment.

---

## 7. Test Set Protocol

The test dataset contains 400 images.

It remains completely separate from the training and validation subsets.

The test set is reserved for final model evaluation and is not used for:

* model selection
* hyperparameter tuning
* threshold selection
* scheduler decisions
* experiment comparison during development

This separation prevents information from the final evaluation set from influencing model development.

---

## 8. Unclean Dataset

The `unclean` dataset contains 450 images across 9 classes:

```text
ambulance
autobus
kamyun
kamyunet
minibus
neysan
savari
taxi
vanet
```

The additional `neysan` class is not part of the known 8-class training taxonomy.

The `unclean` dataset therefore remains separate from the training and validation data. It will be used in later phases for data-quality analysis, unseen-class analysis, and uncertainty/human-review experiments.

---

## 9. Reproducibility Artifact

The following artifact was generated during this phase:

```text
configs/validation_indices.npy
```

The artifact records the validation subset selected from the cleaned training dataset.

The corresponding split-generation code is:

```text
src/data_split.py
```

---

## 10. Conclusion

The data-splitting protocol was successfully established.

The cleaned training dataset of 400 images was divided into:

* 320 training images
* 80 validation images

The split is stratified and reproducible using `seed=42`.

All 8 classes have identical representation in the training and validation subsets.

The 400-image test dataset remains frozen and isolated from model development.

The 450-image `unclean` dataset remains separate and will be evaluated in later phases.

The resulting protocol provides a fixed and reproducible basis for the CNN baseline and subsequent controlled experiments.
