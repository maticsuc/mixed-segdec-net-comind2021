from tkinter import image_names
from unicodedata import decimal
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
import shutil
import torch
from matplotlib.colors import ListedColormap


def create_folder(folder, exist_ok=True):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST or not exist_ok:
            raise

def save_current_learning_method(save_path):
    """
    Shrani end2end.py, utils.py, models.py datoteke v directory modela.
    """
    create_folder(f"{save_path}/learning_method")
    files = ['end2end.py', 'utils.py', 'models.py']
    for file in files:
        shutil.copy2(file, f"{save_path}/learning_method/{file}")

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

def save_predicted_segmentation(predicted_segmentation, sample_name, run_path):
    save_folder = f"{run_path}/seg_pred"
    if not os.path.exists(save_folder):
        create_folder(save_folder)
    plt.imsave(f"{save_folder}/{sample_name}.png", predicted_segmentation, cmap='gray', vmin=0, vmax=1, dpi=200)

def dice_iou(segmentation_predicted, segmentation_truth, seg_threshold, images=None, image_names=None, run_path=None, decisions=None, is_validation=False, faktor=None, save_images=False):

    results_dice = []
    results_iou = []
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
        if save_images:
            save_folder_seg_pred = f"{run_path}/seg_pred"
            save_folder_seg_pred_bin = f"{run_path}/seg_pred_bin"
            create_folder(save_folder_seg_pred)
            create_folder(save_folder_seg_pred_bin)

    # Za vsak par izračunamo dice in IOU
    for i in range(len(segmentation_predicted)):
        seg_pred = segmentation_predicted[i]
        seg_true_bin = segmentation_truth[i].astype(np.uint8)

        # Naredimo binarne maske s ustreznim thresholdom
        seg_pred_bin = (seg_pred > seg_threshold).astype(np.uint8)

        # Beljenje črnih segmentacij, ki so klasificirane kot razpoke
        if not is_validation and decisions is not None and faktor is not None:
            # Primer klasificiran kot razpoka, segmentacija pa crna - spustimo threshold na max pixel * faktor
            if decisions[i] and seg_pred_bin.max().item() == 0:
                spuscen_threshold = faktor * seg_pred.max().item()
                seg_pred_bin = (seg_pred > spuscen_threshold).astype(np.uint8)
                spusceni_thresholdi.append(image_names[i])          

        # Dice
        result_dice = dice(seg_true_bin, seg_pred_bin)
        results_dice += [result_dice]

        # IOU
        result_iou = iou(seg_true_bin, seg_pred_bin)
        results_iou += [result_iou]

        # Vizualizacija
        if images is not None:
            image = images[i]
            image_name = image_names[i]

            # Shanjevanje slik za diplomo
            if save_images:
                plt.imsave(f"{save_folder_seg_pred}/{image_name}.png", seg_pred, cmap='gray', vmin=0, vmax=1, dpi=200)
                plt.imsave(f"{save_folder_seg_pred_bin}/{image_name}.png", seg_pred_bin, cmap='gray', vmin=0, vmax=1, dpi=200)

            plt.figure()
            plt.clf()

            plt.subplot(1, 5, 1)
            plt.xticks([])
            plt.yticks([])
            plt.title('Image')
            plt.imshow(image)
            plt.xlabel(f"Seg thr: {round(seg_threshold, 5)}")
            
            plt.subplot(1, 5, 2)
            plt.xticks([])
            plt.yticks([])
            plt.title('Groundtruth')
            plt.imshow(seg_true_bin, cmap='gray', vmin=0, vmax=1)
            plt.xlabel(f"Dec out: {decisions[i]}")
            
            plt.subplot(1, 5, 3)
            plt.xticks([])
            plt.yticks([])
            plt.title('Segmentation')
            plt.imshow(seg_pred, cmap='gray', vmin=0, vmax=1)
            plt.xlabel(f"IOU: {round(result_iou.item(), 5)}")
            
            plt.subplot(1, 5, 4)
            plt.xticks([])
            plt.yticks([])
            plt.title('Segmentation\nmask')
            plt.imshow(seg_pred_bin, cmap='gray', vmin=0, vmax=1)
            plt.xlabel(f"Dice: {round(result_dice.item(), 5)}")

            plt.subplot(1, 5, 5)
            plt.xticks([])
            plt.yticks([])
            plt.title('Overlap')
            plt.imshow((seg_pred_bin * 2) + seg_true_bin, cmap=ListedColormap([['black', 'gray', 'red', 'white'][i] for i in np.unique((seg_pred_bin * 2) + seg_true_bin)]))
            plt.xlabel(f"Dice: {round(result_dice.item(), 5)}")

            plt.savefig(f"{save_folder}/{round(result_dice.item(), 5):.3f}_dice_{image_name}.png", bbox_inches='tight', dpi=300)
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
    return np.mean(results_dice), np.std(results_dice), np.mean(results_iou), np.std(results_iou), mean_faktor

def segmentation_metrics(seg_truth, seg_predicted, two_pixel_threshold, samples=None, run_path=None, pxl_distance=2, adjusted_thresholds=None):
    # Save folder
    if run_path is not None:
        save_folder = f"{run_path}/seg_metrics"
        create_folder(save_folder)

    n_samples = len(seg_truth)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1 + pxl_distance * 2, 1 + pxl_distance * 2))
    results = []
    
    for i in range(n_samples):

        if adjusted_thresholds is not None:
            two_pixel_threshold = adjusted_thresholds[i]

        y_true = np.array(seg_truth[i]).astype(np.uint8)
        y_true_d = cv2.dilate(y_true, kernel)
        y_pred = (np.array(seg_predicted[i])>two_pixel_threshold).astype(np.uint8)

        tp_d = sum(sum((y_true_d==1)&(y_pred==1))).item()
        fp_d = sum(sum((y_true_d==0)&(y_pred==1))).item()
        fn = sum(sum((y_true==1)&(y_pred==0))).item()

        pr = tp_d / (tp_d + fp_d) if tp_d else 0
        re = tp_d / (tp_d + fn) if tp_d else 0
        f1 = (2 * pr * re) / (pr + re) if pr and re else 0

        results.append((pr, re, f1))

        # Vizualizacija
        if samples is not None:
            image = samples['images'][i]
            image_name = samples['image_names'][i]
            decision = samples['decisions'][i]

            plt.figure()
            plt.clf()

            plt.subplot(1, 5, 1)
            plt.xticks([])
            plt.yticks([])
            plt.title('Image')
            plt.imshow(image)
            plt.xlabel(f"Decision:\n{decision}")

            plt.subplot(1, 5, 2)
            plt.xticks([])
            plt.yticks([])
            plt.title('GT')
            plt.imshow(seg_truth[i], cmap='gray')
            plt.xlabel(f"Seg thr: {round(two_pixel_threshold, 3)}")

            plt.subplot(1, 5, 3)
            plt.xticks([])
            plt.yticks([])
            plt.title('Segmentation')
            plt.imshow(seg_predicted[i], cmap='gray', vmin=0, vmax=1)
            plt.xlabel(f"Pr: {round(pr, 4)}")

            plt.subplot(1, 5, 4)
            plt.xticks([])
            plt.yticks([])
            plt.title('GT\nDilated')
            plt.imshow(y_true_d, cmap='gray', vmin=0, vmax=1)
            plt.xlabel(f"Re: {round(re, 4)}")

            plt.subplot(1, 5, 5)
            plt.xticks([])
            plt.yticks([])
            plt.title('Segmentation\nmask')
            plt.imshow(y_pred, cmap='gray', vmin=0, vmax=1)
            plt.xlabel(f"F1: {round(f1, 4)}")

            plt.savefig(f"{save_folder}/{round(f1, 3):.3f}_{image_name}.png", bbox_inches='tight', dpi=300)
            plt.close()

    pr = np.mean(np.array(results)[:, 0])
    re = np.mean(np.array(results)[:, 1])
    f1 = np.mean(np.array(results)[:, 2])

    return pr, re, f1

# Sccdnet metrics

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def iou(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def precision(y_true, y_pred):
    one = torch.ones_like(torch.Tensor(y_true))
    TP = (y_true * y_pred).sum()
    FP = ((one-y_true)*y_pred).sum()
    return (TP + 1e-15) / (TP + FP + 1e-15)

def recall(y_true, y_pred):
    one = torch.ones_like(torch.Tensor(y_pred))
    one = one.numpy()
    TP = (y_true * y_pred).sum()
    FN = (y_true*(one - y_pred)).sum()
    return (TP + 1e-15) / (TP + FN + 1e-15)