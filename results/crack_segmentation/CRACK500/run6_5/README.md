# Results of training on CRACK500 dataset

[Članek](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9680172)

[Pregled vseh člankov](https://docs.google.com/spreadsheets/d/1AUmJ-JQtpvQt3Rs0maRirAxbBW6zBOBaPq1kVDSdvK0/edit?usp=sharing)

[Dataset](https://github.com/fyangneil/pavement-crack-detection)

## Dataset

| Parameter         | Value       |
| -----------       | ----------- |
| Input channels    | 3           |
| Input height      | 360         |
| Input width       | 640         |
| Train samples     | 1896        |
| Test samples      | 1124        |
| Validation samples| 348         |

| Set         | Positives   | Negatives   |  Sum        |
| ----------- | ----------- | ----------- | ----------- |
| Train       | 1896        | 0           | 1896        |
| Test        | 1124        | 0           | 1124        |
| Validation  | 348         | 0           | 348         |
| **Sum**     | 3368        | 0           | 3368        |

## Run parameters

| Parameter                      | Value       |
| -----------                    | ----------- |
| Batch size                     | 1           |
| Epochs                         | 100         |
| Learning rate                  | 0.001       |
| Delta CLS Loss                 | 0.01        |
| Dilate                         | 1           |
| Dynamically balanced loss      | True        |
| Gradien-flow adjustment        | True        |
| Frequency-of-use sampling      | False       |
| Weighted segmentation loss     | False       |
| Use best model                 | True        |
| Best model type                | seg         |
| Validate                       | True        |
| Validate on test               | False       |
| Augmentation                   | True        |
| Optimizer                      | Adam        |
| Use negatives                  | True        |

### Decision
| Precision | Recall   | F1       | Accuracy | Threshold | TP   | FP   | FN   | TN   |
| ----------| ---------| ---------|----------|-----------|------|------|------|------|
| 1         | 0.996441 | 0.998217 | 0.996441 | 0.925949  | 1120 | 0    | 4    | 0    |

### 2 pixel distance
| Precision     | Recall       | F1           | Threshold |
| --------------| -------------| -------------|-----------|
| 0.743568      | 0.810500     | 0.758308     | 0.305     |

## Losses

![Loss](./loss.png)

![Dec Loss](./loss_dec.png)

![Seg Loss](./loss_seg.png)

![Val Loss](./loss_val.png)

## Scores

![Scores](./scores.png)

### Output

[Output](./crack500_run6_5.out)