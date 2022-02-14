# Results of training on crack_segmentation dataset

## Model architecture

### Run 1

![run1_architecture](./upsampling_7/arhitektura.png)

### Run 2

![run2_architecture](./arhitektura_v2.png)

## Dataset

| Parameter         | Value       |
| -----------       | ----------- |
| Input channels    | 3           |
| Input height      | 640         |
| Input width       | 232         |
| Input width       | 232         |
| Train samples     | 7908        |
| Test samples      | 1695        |
| Validation samples| 1695        |
| Segmented samples | 6921        |


| Set         | Positives   | Negatives   |  Sum        |
| ----------- | ----------- | ----------- | ----------- |
| Train       | 6921        | 987         | 7908        |
| Test        | 1483        | 212         | 1695        |
| Validation  | 1483        | 212         | 1695        |
| **Sum**     | 9887        | 1411        | 11298       |

## Run parameters

| Parameter                      | Value       |
| -----------                    | ----------- |
| Batch size                     | 1           |
| Epochs                         | 50          |
| Learning rate                  | 1           |
| Delta CLS Loss                 | 0.01        |
| Dilate                         | 1           |
| Dynamically balanced loss      | True        |
| Frequency-of-use sampling      | True        |
| Gradien-flow adjustment        | True        |
| Weighted segmentation loss     | False       |
| Use best model                 | True        |
| Validate                       | True        |
| Validate on test               | False       |
| Dice threshold                 | 2           |
| Dice threshold factor          | 1           |

## Test Evaluation

Run 1: EVAL on TEST AUC=0.999352, and AP=0.999905, w/ best thr=0.095512 at f-m=0.996 and FP=10, FN=3
Run 2: EVAL on TEST AUC=0.998385, and AP=0.999756, w/ best thr=0.194860 at f-m=0.996 and FP=5, FN=7

## ROC

Run 1                             |  Run 2
:--------------------------------:|:-------------------------:
![ROC](./upsampling_7/ROC.png)    |  ![ROC](./ROC.png)

## Precision Recall

Run 1                             |  Run 2
:--------------------------------:|:-------------------------:
![ROC](./upsampling_7/precision-recall.png)    |  ![ROC](./precision-recall.png)

## Losses

### Loss Segmentation

Run 1                             |  Run 2
:--------------------------------:|:-------------------------:
![ROC](./upsampling_7/loss_seg.png)    |  ![ROC](./loss_seg.png)

### Loss Decision

Run 1                             |  Run 2
:--------------------------------:|:-------------------------:
![ROC](./upsampling_7/loss_dec.png)    |  ![ROC](./loss_dec.png)


### Total Loss

Run 1                             |  Run 2
:--------------------------------:|:-------------------------:
![ROC](./upsampling_7/loss.png)    |  ![ROC](./loss.png)

### Validation

Run 1                             |  Run 2
:--------------------------------:|:-------------------------:
![ROC](./upsampling_7/loss_val.png)    |  ![ROC](./loss_val.png)

### Dice and IoU
Run 1                             |  Run 2
:--------------------------------:|:-------------------------:
![ROC](./upsampling_7/dice_iou.png)    |  ![ROC](./dice_iou.png)

## Dices
### Run 1
Threshold = 0.1273 (From validation)
|             | mean        | std         |
| ----------- | ----------- | ----------- |
| **Dice**    | 0.6686      | 0.2256      |
| **IoU**     | 0.5409      | 0.2352      |

### Run 2
Threshold = 0.1362 (From validation)
|             | mean        | std         |
| ----------- | ----------- | ----------- |
| **Dice**    | 0.6796      | 0.2121      |
| **IoU**     | 0.5511      | 0.233       |

Run 1                             |  Run 2
:--------------------------------:|:-------------------------:
![dice_output](./upsampling_7/dices/0.413_dice_43.png)     |  ![dice_output](./dices/0.477_dice_43.png)
![dice_output](./upsampling_7/dices/0.674_dice_1039.png)    |  ![dice_output](./dices/0.682_dice_1039.png)
![dice_output](./upsampling_7/dices/0.705_dice_946.png)     |  ![dice_output](./dices/0.681_dice_946.png)
![dice_output](./upsampling_7/dices/0.944_dice_566.png)     |  ![dice_output](./dices/0.888_dice_566.png)
![dice_output](./upsampling_7/dices/0.755_dice_232.png)     |  ![dice_output](./dices/0.784_dice_232.png)
![dice_output](./upsampling_7/dices/0.908_dice_314.png)     |  ![dice_output](./dices/0.927_dice_314.png)

## Outputs
[Run 1](./upsampling_7/nohup.out)

[Run 2](./upsampling_8.out)