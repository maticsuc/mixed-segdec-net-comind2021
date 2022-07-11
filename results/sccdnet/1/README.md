# Results of training on Sccdnet dataset

[Članek](https://www.mdpi.com/2076-3417/11/11/5074)

[Pregled vseh člankov](https://docs.google.com/spreadsheets/d/1AUmJ-JQtpvQt3Rs0maRirAxbBW6zBOBaPq1kVDSdvK0/edit?usp=sharing)

[Dataset](https://github.com/543630836/SCCDNet_crack)

## Dataset

| Parameter         | Value       |
| -----------       | ----------- |
| Input channels    | 3           |
| Input height      | 448         |
| Input width       | 448         |

| Set         | Positives   | Negatives   | Sum         |
| ----------- | ----------- | ----------- | ----------- |
| Train       | 4935        | 1229        | 6164        |
| Test        | 787         | 218         | 1005        |

## Run parameters

| Parameter                      | Value       |
| -----------                    | ----------- |
| Augmentation                   | True        |
| Batch size                     | 10          |
| BCE Loss weight                | True        |
| Best model type                | both        |
| Optimizer                      | Adam        |
| Learning rate                  | 0.001       |
| Delta CLS Loss                 | 0.1         |
| Epochs                         | 100         |
| Dilate                         | 1           |
| Weighted segmentation loss     | False       |
| Dynamically balanced loss      | False       |
| Gradien-flow adjustment        | True        |
| Frequency-of-use sampling      | False       |
| Validate                       | True        |
| Validate on test               | False       |
| Validataton N epochs           | 2           |
| Use best model                 | True        |

# Datasets

| Set         | Train    | Test     | Opis                                                                    |
| ----------- | -------- | -------- | ----------------------------------------------------------------------- |
| Original    | 6164     | 1005     | Originalen dataset iz članka                                            |
| Opening     | 6164     | 1005     | Opening narejen na originalnem članku                                   |
| Clean       | 2915     | 432      | Odstranjeni slabi primeri iz originalnega dataseta - samo za testiranje |

# Runs

| Set         | Črnenje  | Beljenje |
| ----------- | -------- | -------- |
| Run 1       | True     | 0.9      |
| Run 2       | False    | False    |


### Decision

| Run                  | Precision | Recall   | F1       | Accuracy | TP   | FP   | FN   | TN   | Dataset   |
| ---------------------| ----------| ---------| ---------|----------|------|------|------|------|-----------|
| Run 1 best seg+dec   | 0.998729  | 0.998729 | 0.998729 | 0.998010 | 786  | 1    | 1    | 217  | Opening   |
| Run 1 best seg+dec   | 0.998729  | 0.998729 | 0.998729 | 0.998010 | 786  | 1    | 1    | 217  | Original  |
| Run 1 best seg+dec   | 0.995475  | 1.0      | 0.997732 | 0.997685 | 220  | 1    | 0    | 211  | Clean     |
| Run 1 best seg       | 0.998726  | 0.996188 | 0.997455 | 0.996020 | 784  | 1    | 3    | 217  | Opening   |
| Run 1 best seg       | 0.998726  | 0.996188 | 0.997455 | 0.996020 | 784  | 1    | 3    | 217  | Original  |
| Run 1 best seg       | 0.995455  | 0.995455 | 0.995455 | 0.995370 | 219  | 1    | 1    | 211  | Clean     |
| Run 1 - brez SSE     | 0.998728  | 0.997459 | 0.998093 | 0.997015 | 785  | 1    | 2    | 217  | Opening   |
| Run 2                | 0.998729  | 0.998729 | 0.998729 | 0.998010 | 786  | 1    | 1    | 217  | Opening   |
| Run 2                | 0.991173  | 0.998729 | 0.994937 | 0.992040 | 786  | 7    | 1    | 211  | Original  |

### Segmentation

| Run                  | Precision | Recall   | F1       | Dice     | IoU      | Dataset  |
| ---------------------| ----------| ---------| ---------|----------|----------|----------|
| SCCDNet Članek       | 0.7294    | 0.8296   | 0.7763   | 0.7541   | 0.6402   | Original |
| Run 1 best seg+dec   | 0.761628  | 0.861588 | 0.808530 | 0.797259 | 0.692042 | Opening  |
| Run 1 best seg+dec   | 0.748002  | 0.775889 | 0.761690 | 0.743794 | 0.622765 | Original |
| Run 1 best seg+dec   | 0.770487  | 0.908299 | 0.833736 | 0.812214 | 0.743119 | Clean    |
| Run 1 best seg       | 0.784680  | 0.865563 | 0.823140 | 0.812680 | 0.713362 | Opening  |
| Run 1 best seg       | 0.765339  | 0.771863 | 0.768588 | 0.750971 | 0.630939 | Original |
| Run 1 best seg       | 0.780763  | 0.906779 | 0.839066 | 0.819631 | 0.751020 | Clean    |
| Run 1 - brez SSE     | 0.776441  | 0.837102 | 0.805632 | 0.792951 | 0.689838 | Opening  |
| Run 2                | 0.773732  | 0.843309 | 0.807024 | 0.785855 | 0.685849 | Opening  |
| Run 2                | 0.727615  | 0.758179 | 0.742583 | 0.700874 | 0.580615 | Original |

# 5-fold cross validation

Črnenje + Beljenje

| Run      | FP      | FN       | F1       | Dice     | IoU      | Dataset  |
| ---------| --------| ---------| ---------|----------|----------|----------|
| Fold 1   | 8       | 5        | 0.784687 | 0.753528 | 0.649928 | Opening  |
| Fold 2   | 3       | 5        | 0.796758 | 0.768059 | 0.669756 | Opening  |
| Fold 3   | 8       | 4        | 0.800613 | 0.776133 | 0.674688 | Opening  |
| Fold 4   | 9       | 1        | 0.806646 | 0.782884 | 0.683474 | Opening  |
| Fold 5   | 13      | 5        | 0.792372 | 0.766026 | 0.663960 | Opening  |
| Fold 1   | 8       | 5        | 0.729887 | 0.694900 | 0.574542 | Original |
| Fold 2   | 3       | 5        | 0.736729 | 0.695658 | 0.575168 | Original |
| Fold 3   | 8       | 4        | 0.743013 | 0.709577 | 0.589327 | Original |
| Fold 4   | 9       | 1        | 0.745337 | 0.709324 | 0.590062 | Original |
| Fold 5   | 13      | 5        | 0.735733 | 0.698533 | 0.578679 | Original |


| Run          | m-F1     | m-Dice   | m-IoU    | Dataset  |
| -------------| ---------|----------|----------|----------|
| 5-Fold Cross | 0.796215 | 0.769326 | 0.668361 | Opening  |
| 5-Fold Cross | 0.738140 | 0.701598 | 0.581556 | Original |

| Seed | Precision | Recall   | F1       | Dice     | IoU      | FP | FN | Dataset | Blacked segmentations | Adjusted thresholds |
| -----| ----------| ---------| ---------|----------|----------|----|----|---------|-----------------------|---------------------|
| 101  | 0.792487  | 0.863514 | 0.826477 | 0.816748 | 0.718019 | 0  | 2  | Opening | 220                   | 1                   |
| 102  | 0.784680  | 0.865563 | 0.823140 | 0.812680 | 0.713362 | 1  | 3  | Opening | 220                   | 0                   |
| 103  | 0.790072  | 0.856539 | 0.821964 | 0.812904 | 0.713713 | 0  | 3  | Opening | 221                   | 0                   |
| 104  | 0.775150  | 0.868448 | 0.819151 | 0.810403 | 0.709515 | 1  | 2  | Opening | 219                   | 0                   |
| 105  | 0.780280  | 0.864093 | 0.820050 | 0.810717 | 0.711674 | 0  | 2  | Opening | 220                   | 1                   |
| mean | 0.784534  | 0.863631 | 0.822156 | 0.812690 | 0.713257 |    |    |         |                       |                     |
