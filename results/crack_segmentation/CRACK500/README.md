# Results of training on CRACK500 dataset

## Model architecture

![architecture](./arhitektura_v2.png)

## Dataset

| Parameter         | Value       |
| -----------       | ----------- |
| Input channels    | 3           |
| Input height      | 640         |
| Input width       | 360         |
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
| Epochs                         | 50          |
| Learning rate                  | 1.0         |
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

## Razlike med poganjanji

### Run 1
- Počrni segmentacije
- Threshold step: 0.01 - za računanje najboljšega thresholda za F1 pri 2 pxl distance

### Run 2
- Ne počrni segmentacij
- Threshold step: 0.005 - za računanje najboljšega thresholda za F1 pri 2 pxl distance

### Run 3
- Dodana augmentacija
    - Horizontalen in vertikalen flip, rotacija 180, Color Jittering

## Test Evaluation

### Decision
| Run    | Precision | Recall   | F1       | Accuracy | Threshold | TP   | FP   | FN   | TN   |
| -------| ----------| ---------| ---------|----------|-----------|------|------|------|------|
| Run 1  | 1         | 0.997331 | 0.998664 | 0.997331 | 0.999685  | 1121 | 0    | 3    | 0    |
| Run 2  | 1         | 0.996441 | 0.998217 | 0.996641 | 0.999059  | 1120 | 0    | 4    | 0    |
| Run 3  | 1         | 0.997331 | 0.998664 | 0.997331 | 0.999777  | 1121 | 0    | 3    | 0    |

### Segmentation
| Run    | Dice mean | Dice std | IoU mean | IoU std  | Threshold |
| -------| ----------| ---------| ---------|----------|-----------|
| Run 1  | 0.68227   | 0.15953  | 0.53757  | 0.16587  | 0.42      |
| Run 2  | 0.68659   | 0.15140  | 0.54100  | 0.16026  | 0.54      |
| Run 3  | 0.69214   | 0.15253  | 0.54779  | 0.16129  | 0.45      |

### 2 pixel distance
| Run    | Precision     | Recall       | F1           | Threshold |
| -------| --------------| -------------| -------------|-----------|
| Run 1  | 0.728432      | 0.793387     | 0.740356     | 0.4       |
| Run 2  | **0.746946**  | 0.786900     | 0.747593     | 0.52      |
| Run 2  | 0.728117      | **0.814876** | **0.750255** | 0.42      |

### Primerjava - 2 pixel distance

| **Methods**  | Precision  | Recall     | F1         |
| -------------| -----------| -----------| -----------|
| Jenkins      | 0.6811     | 0.6629     | 0.6788     |
| Zhang        | 0.7368     | 0.7165     | 0.7295     |
| Lau          | 0.7426     | 0.7285     | 0.7327     |
| Li (Članek)  | **0.7909** | 0.7650     | **0.7778** |
| **Run 1**    | 0.7284     | **0.7934** | 0.7404     |
| **Run 2**    | 0.7469     | 0.7869     | 0.7476     |
| **Run 3**    | 0.7281     | 0.8149     | 0.7503     |

[Članek](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9680172)

## Losses

| **Loss**          | Run 1                                | Run 2                            | Run 3                            | 
| ------------------| -------------------------------------| ---------------------------------| ---------------------------------|
| Segmentation Loss | ![loss_seg](./run1/loss_seg.png)     | ![loss_seg](./run2/loss_seg.png) | ![loss_seg](./run3/loss_seg.png) |
| Decision Loss     | ![loss_dec](./run1/loss_dec.png)     | ![loss_dec](./run2/loss_dec.png) | ![loss_dec](./run3/loss_dec.png) |
| Total Loss        | ![loss_dec](./run1/loss.png)         | ![loss_dec](./run2/loss.png)     | ![loss_dec](./run3/loss.png)     |
| Validation Loss   | ![loss_dec](./run1/loss_val.png)     | ![loss_dec](./run2/loss_val.png) | ![loss_dec](./run3/loss_val.png) |

### Pr, Re, F1

| **Loss**          | Run 1                                | Run 2                            | Run 3                            | 
| ------------------| -------------------------------------| ---------------------------------| ---------------------------------|
| Scores            | ![loss_seg](./run1/scores.png)       | ![loss_seg](./run2/scores.png)   | ![loss_seg](./run3/scores.png)   |

### Outputs of model learning
[Run 1](./run1/crack500_run1.out)

[Run 2](./run2/crack500_run2.out)

[Run 3](./run3/crack500_run3.out)