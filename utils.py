import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd
import os
import errno
import pickle
import cv2


def create_folder(folder, exist_ok=True):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST or not exist_ok:
            raise


def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP


def plot_sample(image_name, image, segmentation, label, save_dir, decision=None, blur=True, plot_seg=False):
    plt.figure()
    plt.clf()
    plt.subplot(1, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Input image')
    if image.shape[0] < image.shape[1]:
        image = np.transpose(image, axes=[1, 0, 2])
        segmentation = np.transpose(segmentation)
        label = np.transpose(label)
    if image.shape[2] == 1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.subplot(1, 4, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Groundtruth')
    plt.imshow(label, cmap="gray")

    plt.subplot(1, 4, 3)
    plt.xticks([])
    plt.yticks([])
    if decision is None:
        plt.title('Output')
    else:
        plt.title(f"Output:\n{decision:.5f}")
    # display max
    vmax_value = max(1, np.max(segmentation))
    plt.imshow(segmentation, cmap="jet", vmax=vmax_value)

    plt.subplot(1, 4, 4)
    plt.xticks([])
    plt.yticks([])
    plt.title('Output\nscaled')
    if blur:
        normed = segmentation / segmentation.max() if segmentation.max() > 0 else segmentation
        blured = cv2.blur(normed, (32, 32))
        plt.imshow(((blured / blured.max() if blured.max() > 0 else blured) * 255).astype(np.uint8), cmap="jet")
    else:
        plt.imshow(((segmentation / segmentation.max() if segmentation.max() > 0 else segmentation) * 255).astype(np.uint8), cmap="jet")

    out_prefix = '{:.3f}_'.format(decision) if decision is not None else ''
    plt.savefig(f"{save_dir}/{out_prefix}result_{image_name}.jpg", bbox_inches='tight', dpi=300)
    plt.close()

    if plot_seg:
        jet_seg = cv2.applyColorMap((segmentation * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/{out_prefix}_segmentation_{image_name}.png", jet_seg)


def evaluate_metrics(samples, results_path, run_name, segmentation_predicted, segmentation_truth, images, dice_threshold, dataset_kind):
    samples = np.array(samples)

    img_names = samples[:, 4]
    predictions = samples[:, 0]
    labels = samples[:, 3].astype(np.float32)

    metrics = get_metrics(labels, predictions)
    dice_mean, dice_std, iou_mean, iou_std = dice_iou(segmentation_predicted, segmentation_truth, dice_threshold, images, img_names, results_path)

    df = pd.DataFrame(
        data={'prediction': predictions,
              'decision': metrics['decisions'],
              'ground_truth': labels,
              'img_name': img_names})
    df.to_csv(os.path.join(results_path, 'results.csv'), index=False)

    print(
        f'{run_name} EVAL on {dataset_kind} AUC={metrics["AUC"]:f}, and AP={metrics["AP"]:f}, w/ best thr={metrics["best_thr"]:f} at f-m={metrics["best_f_measure"]:.3f} and FP={sum(metrics["FP"]):d}, FN={sum(metrics["FN"]):d}\nDice: mean: {dice_mean:f}, std: {dice_std:f}, IOU: mean: {iou_mean:f}, std: {iou_std:f}, Dice Threshold: {dice_threshold:f}')

    with open(os.path.join(results_path, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
        f.close()

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['recall'], metrics['precision'])
    plt.title('Average Precision=%.4f' % metrics['AP'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f"{results_path}/precision-recall", bbox_inches='tight', dpi=200)

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['FPR'], metrics['TPR'])
    plt.title('AUC=%.4f' % metrics['AUC'])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(f"{results_path}/ROC", bbox_inches='tight', dpi=200)


def get_metrics(labels, predictions):
    metrics = {}
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['thresholds'] = thresholds
    f_measures = 2 * np.multiply(recall, precision) / (recall + precision + 1e-8)
    metrics['f_measures'] = f_measures
    ix_best = np.argmax(f_measures)
    metrics['ix_best'] = ix_best
    best_f_measure = f_measures[ix_best]
    metrics['best_f_measure'] = best_f_measure
    best_thr = thresholds[ix_best]
    metrics['best_thr'] = best_thr
    FPR, TPR, _ = roc_curve(labels, predictions)
    metrics['FPR'] = FPR
    metrics['TPR'] = TPR
    AUC = auc(FPR, TPR)
    metrics['AUC'] = AUC
    AP = auc(recall, precision)
    metrics['AP'] = AP
    decisions = predictions >= best_thr
    metrics['decisions'] = decisions
    FP, FN, TN, TP = calc_confusion_mat(decisions, labels)
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TN'] = TN
    metrics['TP'] = TP
    metrics['accuracy'] = (sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN))
    return metrics

def dice_iou(segmentation_predicted, segmentation_truth, seg_threshold, images=None, image_names=None, run_path=None, decisions=None, is_validation=False, faktor=None):

    results_dice = []
    result_iou = []
    spusceni_thresholdi = []
    faktorji_spustitve_thresholdov = []
    mean_faktor = None

    # Preverimo ali so vsi listi enako dolgi
    if images is not None:
        if not (len(segmentation_predicted) == len(segmentation_truth) == len(images) == len(image_names)):
            raise ValueError('Not equal size of segmentation masks or images')
    else:
        if not (len(segmentation_predicted) == len(segmentation_truth)):
            raise ValueError('Not equal size of segmentation masks')
    
    # Save folder
    if run_path is not None:
        save_folder = f"{run_path}/dices"
        create_folder(save_folder)

    # Za vsak par izračunamo dice in IOU
    for i in range(len(segmentation_predicted)):
        seg_pred = segmentation_predicted[i]
        seg_true_bin = segmentation_truth[i]

        # Naredimo binarne maske s ustreznim thresholdom
        seg_pred_bin = (seg_pred > seg_threshold).astype(np.uint8)

        # Beljenje črnih segmentacij, ki so klasificirane kot razpoke
        if not is_validation and decisions is not None and faktor is not None:
            # Primer klasificiran kot razpoka, segmentacija pa crna - spustimo threshold na max pixel * faktor
            if decisions[i] and seg_pred_bin.max().item() == 0:
                spuscen_threshold = faktor * seg_pred.max().item()
                seg_pred_bin = (seg_pred > spuscen_threshold).astype(np.uint8)
                spusceni_thresholdi.append(image_names[i])

        # Računanje najboljšega faktorja za zmanjševanje thresholda (VALIDACIJA)
        if is_validation and decisions is not None:
            # Primer klasificiran kot razpoka, segmentacija pa crna - spustimo threshold na max pixel * faktor
            if decisions[i] and seg_pred_bin.max().item() == 0:
                best_dice = -1
                best_faktor = -1
                step = 0.01
                for i in np.arange(0.01, 1, step):
                    i = round(i, 2)
                    spuscen_threshold = i * seg_pred.max().item()
                    seg_pred_bin = (seg_pred > spuscen_threshold).astype(np.uint8)
                    dice = (2 * (seg_true_bin * seg_pred_bin).sum() + 1e-15) / (seg_true_bin.sum() + seg_pred_bin.sum() + 1e-15)
                    if dice > best_dice:
                        best_dice = dice
                        best_faktor = i

            faktorji_spustitve_thresholdov.append(best_faktor)           

        # Dice
        dice = (2 * (seg_true_bin * seg_pred_bin).sum() + 1e-15) / (seg_true_bin.sum() + seg_pred_bin.sum() + 1e-15)
        results_dice += [dice]

        # IOU
        intersection = (seg_pred_bin * seg_true_bin).sum()
        union = seg_pred_bin.sum() + seg_true_bin.sum() - intersection
        iou = (intersection + 1e-15) / (union + 1e-15)
        result_iou += [iou]

        # Vizualizacija
        if images is not None:
            image = images[i]
            image_name = image_names[i]
            plt.figure()
            plt.clf()

            plt.subplot(1, 4, 1)
            plt.xticks([])
            plt.yticks([])
            plt.title('Image')
            plt.imshow(image)
            plt.xlabel(f"Seg thr: {seg_threshold:f}")
            
            plt.subplot(1, 4, 2)
            plt.xticks([])
            plt.yticks([])
            plt.title('Groundtruth')
            plt.imshow(seg_true_bin, cmap='gray')
            plt.xlabel(f"Dec out: {decisions[i]}")
            
            plt.subplot(1, 4, 3)
            plt.xticks([])
            plt.yticks([])
            plt.title('Segmentation')
            plt.imshow(seg_pred, cmap='gray', vmin=0, vmax=1) # Popravljeno z vmin in vmax argumenti
            plt.xlabel(f"IOU: {round(iou, 5)}")
            
            plt.subplot(1, 4, 4)
            plt.xticks([])
            plt.yticks([])
            plt.title('Segmentation\nmask')
            plt.imshow(seg_pred_bin, cmap='gray', vmin=0, vmax=1)
            plt.xlabel(f"Dice: {round(dice, 5)}")
            plt.savefig(f"{save_folder}/{round(dice, 3):.3f}_dice_{image_name}.png", bbox_inches='tight', dpi=300)
            plt.close()

    # Zapisem primere s spuscenim thresholdom v txt datoteko
    if len(spusceni_thresholdi) > 0:
        txt_file = "spusceni_thresholdi.txt"
        file = open(os.path.join(run_path, txt_file), "w")
        for sample in spusceni_thresholdi:
            file.write(sample + "\n")
        file.close()

    if len(faktorji_spustitve_thresholdov) > 0:
        mean_faktor = np.mean(np.array(faktorji_spustitve_thresholdov))

    # Vrnemo povprečno vrednost ter standardno deviacijo za dice in IOU
    return np.mean(results_dice), np.std(results_dice), np.mean(result_iou), np.std(result_iou), mean_faktor