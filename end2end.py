import sched
from tabnanny import verbose
import matplotlib
from sklearn import metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import SegDecNet
import numpy as np
import os
from torch import nn as nn
import torch
import utils
import pandas as pd
from data.dataset_catalog import get_dataset
import random
import cv2
from config import Config
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score
from datetime import datetime

LVL_ERROR = 10
LVL_INFO = 5
LVL_DEBUG = 1

LOG = 1  # Will log all mesages with lvl greater than this
SAVE_LOG = True

WRITE_TENSORBOARD = False


class End2End:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.storage_path: str = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)

    def _log(self, message, lvl=LVL_INFO):
        time = datetime.now().strftime("%d-%m-%y %H:%M")
        n_msg = f"{time} {self.run_name} {message}"
        if lvl >= LOG:
            print(n_msg)

    def train(self):
        self._set_results_path()
        self._create_results_dirs()
        self.print_run_params()
        if self.cfg.REPRODUCIBLE_RUN:
            self._log("Reproducible run, fixing all seeds to:1337", LVL_DEBUG)
            np.random.seed(1337)
            torch.manual_seed(1337)
            random.seed(1337)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = self._get_device()
        model = self._get_model().to(device)
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        loss_seg, loss_dec = self._get_loss(True), self._get_loss(False)

        train_loader = get_dataset("TRAIN", self.cfg)
        validation_loader = get_dataset("VAL", self.cfg)

        tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path) if WRITE_TENSORBOARD else None

        # Save current learning method to model's directory
        utils.save_current_learning_method(save_path=self.run_path)

        losses, validation_data, best_model_metrics, validation_metrics, lrs = self._train_model(device, model, train_loader, loss_seg, loss_dec, optimizer, scheduler, validation_loader, tensorboard_writer)
        train_results = (losses, validation_data, validation_metrics, lrs)
        self._save_train_results(train_results)
        self._save_model(model)

        self.eval(model=model, device=device, save_images=self.cfg.SAVE_IMAGES, plot_seg=False, reload_final=False, dice_threshold=best_model_metrics['dice_threshold'], best_model_metrics=best_model_metrics)

        self._save_params()

    def eval(self, model, device, save_images, plot_seg, reload_final, dice_threshold, eval_loader=None, best_model_metrics=None):
        self.reload_model(model, reload_final)
        is_validation = True
        if eval_loader is None:
            eval_loader = get_dataset("TEST", self.cfg)
            is_validation = False
        self.eval_model(device, model, eval_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=is_validation, plot_seg=plot_seg, dice_threshold=dice_threshold, dec_threshold=best_model_metrics['dec_threshold'], two_pxl_threshold=best_model_metrics['two_pxl_threshold'])

    def training_iteration(self, data, device, model, criterion_seg, criterion_dec, optimizer, weight_loss_seg, weight_loss_dec,
                           tensorboard_writer, iter_index):
        images, seg_masks, is_segmented, sample_names, is_pos = data

        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for >1

        num_subiters = int(batch_size / memory_fit)

        total_loss = 0
        total_correct = 0

        optimizer.zero_grad()

        total_loss_seg = 0
        total_loss_dec = 0

        for sub_iter in range(num_subiters):
            images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_mask_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            is_pos_ = seg_mask_.max().reshape((memory_fit, 1)).to(device)

            if tensorboard_writer is not None and iter_index % 100 == 0:
                tensorboard_writer.add_image(f"{iter_index}/image", images_[0, :, :, :])

            decision, seg_mask_predicted = model(images_)

            if is_segmented[sub_iter]:
                loss_seg = criterion_seg(seg_mask_predicted, seg_mask_)
                loss_dec = criterion_dec(decision, is_pos_)

                total_loss_seg += loss_seg.item()
                total_loss_dec += loss_dec.item()

                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_seg * loss_seg + weight_loss_dec * loss_dec
            else:
                loss_dec = criterion_dec(decision, is_pos_)
                total_loss_dec += loss_dec.item()

                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_dec * loss_dec

            total_loss += loss.item()

            loss.backward()

        # Backward and optimize
        optimizer.step()
        optimizer.zero_grad()

        return total_loss_seg, total_loss_dec, total_loss, total_correct

    def _train_model(self, device, model, train_loader, criterion_seg, criterion_dec, optimizer, scheduler, validation_set, tensorboard_writer):
        losses = []
        validation_data = []
        validation_metrics = []
        lrs = []
        max_validation = -1
        best_dice = -1
        best_f1 = -1
        validation_step = self.cfg.VALIDATION_N_EPOCHS

        num_epochs = self.cfg.EPOCHS
        samples_per_epoch = len(train_loader) * self.cfg.BATCH_SIZE

        self.set_dec_gradient_multiplier(model, 0.0)

        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                self._save_model(model, f"ep_{epoch:02}.pth")

            model.train()

            weight_loss_seg, weight_loss_dec = self.get_loss_weights(epoch)
            dec_gradient_multiplier = self.get_dec_gradient_multiplier()
            self.set_dec_gradient_multiplier(model, dec_gradient_multiplier)

            epoch_loss_seg, epoch_loss_dec, epoch_loss = 0, 0, 0
            epoch_correct = 0
            from timeit import default_timer as timer

            time_acc = 0
            start = timer()
            for iter_index, (data) in enumerate(train_loader):
                start_1 = timer()
                curr_loss_seg, curr_loss_dec, curr_loss, correct = self.training_iteration(data, device, model,
                                                                                           criterion_seg,
                                                                                           criterion_dec,
                                                                                           optimizer, weight_loss_seg,
                                                                                           weight_loss_dec,
                                                                                           tensorboard_writer, (epoch * samples_per_epoch + iter_index))

                end_1 = timer()
                time_acc = time_acc + (end_1 - start_1)

                epoch_loss_seg += curr_loss_seg
                epoch_loss_dec += curr_loss_dec
                epoch_loss += curr_loss

                epoch_correct += correct

            end = timer()

            epoch_loss_seg = epoch_loss_seg / samples_per_epoch
            epoch_loss_dec = epoch_loss_dec / samples_per_epoch
            epoch_loss = epoch_loss / samples_per_epoch
            losses.append((epoch_loss_seg, epoch_loss_dec, epoch_loss, epoch))

            self._log(f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}, correct={epoch_correct}/{samples_per_epoch}, in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

            scheduler.step()
            last_learning_rate = scheduler.get_last_lr()[-1]
            self._log(f"Last computing learning rate by scheduler: {last_learning_rate}")
            lrs.append((epoch, last_learning_rate))

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/Train/segmentation", epoch_loss_seg, epoch)
                tensorboard_writer.add_scalar("Loss/Train/classification", epoch_loss_dec, epoch)
                tensorboard_writer.add_scalar("Loss/Train/joined", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Accuracy/Train/", epoch_correct / samples_per_epoch, epoch)

            if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1):
                validation_ap, validation_accuracy, val_metrics = self.eval_model(device=device, model=model, eval_loader=validation_set, save_folder=None, save_images=False, is_validation=True, plot_seg=False, dice_threshold=None)
                validation_data.append((validation_ap, epoch))
                validation_metrics.append((epoch, val_metrics))

                if self.cfg.BEST_MODEL_TYPE == "dec" and validation_ap > max_validation:
                    self._save_model(model, "best_state_dict.pth")
                    max_validation = validation_ap

                elif self.cfg.BEST_MODEL_TYPE == "seg" and val_metrics['F1'] > best_f1:
                    self._log(f"New best model based on {self.cfg.BEST_MODEL_TYPE} metrics.")
                    self._save_model(model, "best_state_dict.pth")
                    best_model_metrics = val_metrics
                    best_f1 = val_metrics['F1']
                
                elif self.cfg.BEST_MODEL_TYPE == "both" and ((validation_ap > max_validation and val_metrics['dice_score'] >= best_dice) or (val_metrics['dice_score'] > best_dice and validation_ap >= max_validation)):
                    self._save_model(model, "best_state_dict.pth")
                    max_validation = validation_ap
                    best_dice = val_metrics['dice_score']
                    best_model_metrics = val_metrics

                model.train()
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("Accuracy/Validation/", validation_accuracy, epoch)

        return losses, validation_data, best_model_metrics, validation_metrics, lrs

    def eval_model(self, device, model, eval_loader, save_folder, save_images, is_validation, plot_seg, dice_threshold, dec_threshold=None, two_pxl_threshold=None, faktor=None):
        model.eval()

        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT

        res = []
        predictions, predictions_truths = [], []

        images, predicted_segs, true_segs = [], [], []
        predicted_segs_pos, predicted_segs_neg = [], []
        samples = {"images": list(), "image_names": list()}

        for data_point in eval_loader:
            image, seg_mask, _, sample_name, is_pos = data_point
            image, seg_mask = image.to(device), seg_mask.to(device)
            #is_pos = (seg_mask.max() > 0).reshape((1, 1)).to(device).item()
            is_pos = is_pos.item()
            prediction, seg_mask_predicted = model(image)
            prediction = nn.Sigmoid()(prediction)
            seg_mask_predicted = nn.Sigmoid()(seg_mask_predicted)

            prediction = prediction.item()
            image = image.detach().cpu().numpy()
            seg_mask = seg_mask.detach().cpu().numpy()
            seg_mask_predicted = seg_mask_predicted.detach().cpu().numpy()

            image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            predictions.append(prediction)
            predictions_truths.append(is_pos)
            res.append((prediction, None, None, is_pos, sample_name[0]))

            seg_mask_predicted = seg_mask_predicted[0][0]
            seg_mask = seg_mask[0][0]
            predicted_segs.append(seg_mask_predicted)
            samples["image_names"].append(sample_name[0])
            samples["images"].append(image)
            true_segs.append(seg_mask)
            images.append(image)

            if not is_validation and save_images:
                """
                if dec_threshold and prediction <= dec_threshold:
                    utils.plot_sample(sample_name[0], image, np.zeros(seg_mask_predicted.shape), seg_mask, save_folder, decision=prediction, plot_seg=plot_seg)
                else:
                    utils.plot_sample(sample_name[0], image, seg_mask_predicted, seg_mask, save_folder, decision=prediction, plot_seg=plot_seg)
                """
                utils.plot_sample(sample_name[0], image, seg_mask_predicted, seg_mask, save_folder, decision=prediction, plot_seg=plot_seg)

        if is_validation:
            # Računanje thresholda za decision net
            metrics = utils.get_metrics(np.array(predictions_truths), np.array(predictions))
            FP, FN, TP, TN = list(map(sum, [metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"]]))
            self._log(f"VALIDATION on {eval_loader.dataset.kind} set || AUC={metrics['AUC']:f}, and AP={metrics['AP']:f}, with best thr={metrics['best_thr']:f} sat f-measure={metrics['best_f_measure']:.3f} and FP={FP:d}, FN={FN:d}, TOTAL SAMPLES={FP + FN + TP + TN:d}")
            #val_metrics = dict()
            #val_metrics['dec_threshold'] = metrics['best_thr']

            # Naredim decisions z izračunanim thresholdom
            decisions = np.array(predictions) > metrics['best_thr']

            """
            # Vse predictane non-crack slike pocrnim
            non_crack_seg = np.zeros(predicted_segs[0].shape)
            non_crack_counter = 0
            pocrnjeni_primeri = list()

            for i in range(len(predicted_segs)):
                if not decisions[i]:
                    predicted_segs[i] = non_crack_seg
                    non_crack_counter += 1
                    pocrnjeni_primeri.append(samples[i])
            
            self._log(f"Spremenil {non_crack_counter} segmentacij v crne.")

            # Zapisem pocrnjene primere v txt datoteko
            txt_file = "pocrnjeni_primeri_val.txt"
            file = open(os.path.join(self.run_path, txt_file), "w")
            for sample in pocrnjeni_primeri:
                file.write(sample + "\n")
            file.close()
            """

            # Najboljši F1, Pr, Re, threshold
            val_metrics = self.seg_val_metrics(true_segs, predicted_segs, eval_loader.dataset.kind)
            val_metrics['dec_threshold'] = metrics['best_thr']

            """
            # Računanje dice thresholda
            # 1. Minimum maksimalnih pikslov vseh predikcij
            if self.cfg.DICE_THRESHOLD == 1:
                max_pixels_pos = np.amax(np.amax(np.array(predicted_segs_pos), axis=1), axis=1) # Max piksli pozitivnih predicted segmentacij
                min_pixel_of_max_pixels_pos = max_pixels_pos.min().item() # Min piksel vseh max pikslov
                
                max_pixels_neg = np.amax(np.amax(np.array(predicted_segs_neg), axis=1), axis=1) # Max piksli negativnih predicted segmentacij
                min_pixel_of_max_pixels_neg = max_pixels_neg.min().item() # Min piksel vseh max pikslov
                
                dice_threshold = min_pixel_of_max_pixels_pos
                val_metrics['dice_threshold'] = dice_metrics['best_thr']
            
            # 2. precision_recall, subsampling
            elif self.cfg.DICE_THRESHOLD == 2:
                dice_metrics = utils.get_metrics(np.array(true_segs, dtype=bool).flatten()[::self.cfg.DICE_THR_FACTOR], np.array(predicted_segs).flatten()[::self.cfg.DICE_THR_FACTOR])
                dice_threshold = dice_metrics['best_thr']
                val_metrics['dice_threshold'] = dice_metrics['best_thr']
                val_metrics['dice_score'] = dice_metrics['best_f_measure']
                self._log(f"Dice: {val_metrics['dice_score']}, threshold: {val_metrics['dice_threshold']}")
            
            """
            # Zmanjševanje thresholda
            self._log(f"Racunanje Dice in IoU z razlicnimi segmentacijskimi thresholdi.")
            threshold_decrease_results = dict()
            step = 0.005
            for i in np.arange(0.01, 1, step):
                decreased_threshold = i
                dice_mean, dice_std, iou_mean, iou_std, faktor = utils.dice_iou(segmentation_predicted=predicted_segs, segmentation_truth=true_segs, seg_threshold=decreased_threshold, decisions=decisions, is_validation=True)
                threshold_decrease_results[decreased_threshold] = (dice_mean, dice_std, iou_mean, iou_std, faktor)
            
            best_dice = None
            best_thr = None
            best_faktor = None
            for thr, dice_results in threshold_decrease_results.items():
                if best_dice is None and best_thr is None:
                    best_dice = dice_results[0]
                    best_thr = thr
                if dice_results[0] > best_dice:
                    best_dice = dice_results[0]
                    best_thr = thr
                    best_faktor = dice_results[4]
            
            self._log(f"Best Dice: {best_dice} at threshold: {best_thr}")
            self._log(f"Faktor za zmanjsevanje thresholda glede na max pixel v primerih crne segmentacije: {best_faktor}")
            val_metrics['dice_threshold'] = best_thr
            val_metrics['dice_score'] = best_dice
            val_metrics['faktor'] = best_faktor

            return metrics["AP"], metrics["accuracy"], val_metrics
        else:
            decisions = np.array(predictions) > dec_threshold
            samples["decisions"] = list(decisions)
            FP, FN, TN, TP = utils.calc_confusion_mat(decisions, np.array(predictions_truths))

            fp = sum(FP).item()
            fn = sum(FN).item()
            tn = sum(TN).item()
            tp = sum(TP).item()

            pr = tp / (tp + fp) if tp else 0
            re = tp / (tp + fn) if tp else 0
            f1 = (2 * pr * re) / (pr + re) if pr and re else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            self._log(f"Decision EVAL on {eval_loader.dataset.kind}. Pr: {pr:f}, Re: {re:f}, F1: {f1:f}, Accuracy: {accuracy:f}, Threshold: {dec_threshold}")
            self._log(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

            """
            # Vse predictane non-crack slike pocrnim
            non_crack_seg = np.zeros(predicted_segs[0].shape)
            non_crack_counter = 0
            pocrnjeni_primeri = list()

            for i in range(len(predicted_segs)):
                if not decisions[i]:
                    predicted_segs[i] = non_crack_seg
                    non_crack_counter += 1
                    pocrnjeni_primeri.append(samples[i])
            
            self._log(f"Spremenil {non_crack_counter} segmentacij v crne.")

            # Zapisem pocrnjene primere v txt datoteko
            txt_file = "pocrnjeni_primeri_test.txt"
            file = open(os.path.join(self.run_path, txt_file), "w")
            for sample in pocrnjeni_primeri:
                file.write(sample + "\n")
            file.close()

            """
            dice_mean, dice_std, iou_mean, iou_std, faktor = utils.dice_iou(segmentation_predicted=predicted_segs, segmentation_truth=true_segs, seg_threshold=dice_threshold, images=images, image_names=np.array(res)[:, 4], run_path=self.run_path, decisions=decisions, faktor=None)
            
            self._log(f"Segmentation EVAL on {eval_loader.dataset.kind}. Dice: mean: {dice_mean:f}, std: {dice_std:f}, IOU: mean: {iou_mean:f}, std: {iou_std:f}, Dice Threshold: {dice_threshold:f}")

            # Segmentation metrics + vizualizacija

            self._log(f"Evaluation metrics on {eval_loader.dataset.kind} set. 2 pixel distance used.")
           
            pr, re, f1 = utils.segmentation_metrics(seg_truth=true_segs, seg_predicted=predicted_segs, two_pixel_threshold=two_pxl_threshold, samples=samples, run_path=self.run_path)

            self._log(f"Pr: {pr:f}, Re: {re:f}, F1: {f1:f}, threshold: {two_pxl_threshold}")

    def get_dec_gradient_multiplier(self):
        if self.cfg.GRADIENT_ADJUSTMENT:
            grad_m = 0
        else:
            grad_m = 1

        self._log(f"Returning dec_gradient_multiplier {grad_m}", LVL_DEBUG)
        return grad_m

    def set_dec_gradient_multiplier(self, model, multiplier):
        model.set_gradient_multipliers(multiplier)

    def get_loss_weights(self, epoch):
        total_epochs = float(self.cfg.EPOCHS)

        if self.cfg.DYN_BALANCED_LOSS:
            seg_loss_weight = 1 - (epoch / total_epochs)
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS * (epoch / total_epochs)
        else:
            seg_loss_weight = 1
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS

        self._log(f"Returning seg_loss_weight {seg_loss_weight} and dec_loss_weight {dec_loss_weight}", LVL_DEBUG)
        return seg_loss_weight, dec_loss_weight

    def reload_model(self, model, load_final=False):
        if self.cfg.USE_BEST_MODEL:
            path = os.path.join(self.model_path, "best_state_dict.pth")
            model.load_state_dict(torch.load(path, map_location='cuda:0')) # model.load_state_dict(torch.load(path, map_location='cuda:0'))
            self._log(f"Loading model state from {path}")
        elif load_final:
            path = os.path.join(self.model_path, "final_state_dict.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        else:
            self._log("Keeping same model state")

    def _save_params(self):
        params = self.cfg.get_as_dict()
        params_lines = sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", params.items()))
        fname = os.path.join(self.run_path, "run_params.txt")
        with open(fname, "w+") as f:
            f.writelines(params_lines)

    def _save_train_results(self, results):
        losses, validation_data, validation_metrics, lrs = results
        ls, ld, l, le = map(list, zip(*losses))
        plt.plot(le, l, label="Loss", color="red")
        plt.plot(le, ls, label="Loss seg")
        plt.plot(le, ld, label="Loss dec")
        plt.ylim(bottom=0)
        plt.grid()
        plt.xlabel("Epochs")
        if self.cfg.VALIDATE:
            v, ve = map(list, zip(*validation_data))
            plt.twinx()
            plt.plot(ve, v, label="Validation AP", color="Green")
            plt.ylim((0, 1))
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "loss_val"), dpi=200)

        df_loss = pd.DataFrame(data={"loss_seg": ls, "loss_dec": ld, "loss": l, "epoch": le})
        df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

        if self.cfg.VALIDATE:
            df_loss = pd.DataFrame(data={"validation_data": ls, "loss_dec": ld, "loss": l, "epoch": le})
            df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)
        
        # Dice & IOU plot
        if len(validation_metrics) != 0:
            epochs, metrics = map(list, zip(*validation_metrics))
            f1 = [i['F1'] for i in metrics]
            pr = [i['Pr'] for i in metrics]
            re = [i['Re'] for i in metrics]
            plt.clf()
            plt.plot(epochs, f1, label="F1")
            plt.plot(epochs, pr, label="Pr")
            plt.plot(epochs, re, label="Re")
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.savefig(os.path.join(self.run_path, "scores"), dpi=200)

        # Loss plot
        # Loss
        plt.clf()
        plt.plot(le, l)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.run_path, "loss"), dpi=200)
        
        # Loss Segmentation
        plt.clf()
        plt.plot(le, ls)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Segmentation")
        plt.savefig(os.path.join(self.run_path, "loss_seg"), dpi=200)

        # Loss Dec
        plt.clf()
        plt.plot(le, ld)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Dec")
        plt.savefig(os.path.join(self.run_path, "loss_dec"), dpi=200)

        # Learning rate plot
        epochs, lr = map(list, zip(*lrs))
        plt.clf()
        plt.plot(epochs, lr)
        plt.xlabel("Epochs")
        plt.ylabel("Learning rate")
        plt.savefig(os.path.join(self.run_path, "learning_rate"), dpi=200)

    def _save_model(self, model, name="final_state_dict.pth"):
        output_name = os.path.join(self.model_path, name)
        self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)

        torch.save(model.state_dict(), output_name)

    def _get_optimizer(self, model):
        if self.cfg.OPTIMIZER == "sgd":
            return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE)
        elif self.cfg.OPTIMIZER == "adam":
            return torch.optim.Adam(model.parameters(), self.cfg.LEARNING_RATE)

    def _get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95, verbose=True)

    def _get_loss(self, is_seg):
        reduction = "none" if self.cfg.WEIGHTED_SEG_LOSS and is_seg else "mean"
        return nn.BCEWithLogitsLoss(reduction=reduction).to(self._get_device())

    def _get_device(self):
        return f"cuda:{self.cfg.GPU}"

    def _set_results_path(self):
        self.run_name = f"{self.cfg.RUN_NAME}_FOLD_{self.cfg.FOLD}" if self.cfg.DATASET in ["KSDD", "DAGM"] else self.cfg.RUN_NAME

        results_path = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        self.tensorboard_path = os.path.join(results_path, "tensorboard", self.run_name)

        run_path = os.path.join(results_path, self.cfg.RUN_NAME)
        if self.cfg.DATASET in ["KSDD", "DAGM"]:
            run_path = os.path.join(run_path, f"FOLD_{self.cfg.FOLD}")

        self._log(f"Executing run with path {run_path}")

        self.run_path = run_path
        self.model_path = os.path.join(run_path, "models")
        self.outputs_path = os.path.join(run_path, "test_outputs")

    def _create_results_dirs(self):
        list(map(utils.create_folder, [self.run_path, self.model_path, self.outputs_path, ]))

    def _get_model(self):
        seg_net = SegDecNet(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS)
        return seg_net

    def print_run_params(self):
        for l in sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", self.cfg.get_as_dict().items())):
            k, v = l.split(":")
            self._log(f"{k:25s} : {str(v.strip())}")
        
    def seg_val_metrics(self, truth_segmentations, predicted_segmentations, dataset_kind, threshold_step=0.005, pxl_distance=2):
        n_samples = len(truth_segmentations)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1 + pxl_distance * 2, 1 + pxl_distance * 2))
        thresholds, pr_results, re_results, f1_results = [], [], [], []
        metrics = dict()

        self._log(f"Validation metrics on {dataset_kind} set. {pxl_distance} pixel distance used. Threshold step: {threshold_step}")

        for threshold in np.arange(0.1, 1, threshold_step):
            results = []
            for i in range(n_samples):
                y_true = np.array(truth_segmentations[i]).astype(np.uint8)
                y_true_d = cv2.dilate(y_true, kernel)
                y_pred = (np.array(predicted_segmentations[i])>threshold).astype(np.uint8)

                tp_d = sum(sum((y_true_d==1)&(y_pred==1))).item()
                fp_d = sum(sum((y_true_d==0)&(y_pred==1))).item()
                fn = sum(sum((y_true==1)&(y_pred==0))).item()

                pr = tp_d / (tp_d + fp_d) if tp_d else 0
                re = tp_d / (tp_d + fn) if tp_d else 0
                f1 = (2 * pr * re) / (pr + re) if pr and re else 0

                results.append((pr, re, f1))

            thresholds.append(threshold)
            pr_results.append(np.mean(np.array(results)[:, 0]))
            re_results.append(np.mean(np.array(results)[:, 1]))
            f1_results.append(np.mean(np.array(results)[:, 2]))

        f1_max_index = f1_results.index(max(f1_results))
        metrics['Pr'] = pr_results[f1_max_index]
        metrics['Re'] = re_results[f1_max_index]
        metrics['F1'] = max(f1_results)
        metrics['two_pxl_threshold'] = thresholds[f1_max_index]

        self._log(f"Best F1: {metrics['F1']:f} at {thresholds[f1_max_index]:f}. Pr: {metrics['Pr']:f}, Re: {metrics['Re']:f}")
                
        return metrics