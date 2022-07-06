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
| Fold 1   | 6       | 5        | 0.783203 | 0.742407 | 0.639357 | Opening  |
| Fold 2   | 3       | 5        | 0.796758 | 0.768059 | 0.669756 | Opening  |
| Fold 2   | 3       | 5        | 0.736729 | 0.695658 | 0.575168 | Original |
| Fold 3   | 8       | 4        | 0.800613 | 0.776133 | 0.674688 | Opening  |
| Fold 4   | 8       | 3        | 0.803886 | 0.777554 | 0.678392 | Opening  |
| Fold 4   | 9       | 1        | 0.745337 | 0.709324 | 0.590062 | Original |
| Fold 5   | 8       | 13       | 0.759626 | 0.725968 | 0.615041 | Opening  |
| Fold 5   | 13      | 5        | 0.735733 | 0.698533 | 0.578679 | Original |
