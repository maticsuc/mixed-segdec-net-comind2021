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

| Set         | Positives   | Negatives   |
| ----------- | ----------- | ----------- |
| Train       | 1896        | 0           |
| Test        | 1124        | 0           |
| Validation  | 348         | 0           |

## Run parameters

Za vse run-e so bili uporabljeni parametri.

| Parameter                      | Value       |
| -----------                    | ----------- |
| Optimizer                      | Adam        |
| Learning rate                  | 0.001       |
| Delta CLS Loss                 | 0.01        |
| Epochs                         | 100         |
| Batch size                     | 10          |
| Dilate                         | 1           |
| Weighted segmentation loss     | False       |
| Dynamically balanced loss      | True        |
| Gradien-flow adjustment        | True        |
| Frequency-of-use sampling      | False       |
| Validate                       | True        |
| Validate on test               | False       |
| Validataton N epochs           | 2           |
| Use best model                 | True        |
| Best model type                | seg         |
| Augmentation                   | True        |

### Decision

| Run         | Precision | Recall   | F1       | Accuracy | TP   | FP   | FN   | TN   |
| ------------| ----------| ---------| ---------|----------|------|------|------|------|
| 1           | 1.0       | 0.994662 | 0.997324 | 0.994662 | 1118 | 0    | 6    | 0    |
| 2           | 1.0       | 0.998221 | 0.999110 | 0.998221 | 1122 | 0    | 2    | 0    |
| 3           | 1.0       | 1.0      | 1.0      | 1.0      | 1124 | 0    | 0    | 0    |
| 4           | 1.0       | 0.997331 | 0.998664 | 0.997331 | 1121 | 0    | 3    | 0    |

### Segmentation 2 pixel distance

| Run         | Precision     | Recall       | F1           | Threshold | Train Negatives |
| ------------| --------------| -------------| -------------|-----------|-----------------|
| 1           | 0.748412      | 0.817055     | 0.762888     | 0.46      | 757             |
| 2           | 0.756516      | 0.793254     | 0.754537     | 0.31      | 757             |
| 3           | 0.736412      | 0.804229     | 0.746711     | 0.305     | 757             |
| 4           | 0.739240      | 0.811107     | 0.755910     | 0.38     | 757             |

### Outputs

[1](./crack500_run9_1.out)

[2](./crack500_run9_1_1.out)

[3](./crack500_run9_1_2.out)

[4](./crack500_run9_1_3.out)