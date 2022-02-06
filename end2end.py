import matplotlib

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
        loss_seg, loss_dec = self._get_loss(True), self._get_loss(False)

        # Nov segmentacijski loss - BCEWithLogitsLoss
        loss_seg_upsampled = self._get_loss(True)

        train_loader = get_dataset("TRAIN", self.cfg)
        validation_loader = get_dataset("VAL", self.cfg)

        tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path) if WRITE_TENSORBOARD else None

        losses, validation_data, dice_threshold, dices_iou = self._train_model(device, model, train_loader, loss_seg, loss_seg_upsampled, loss_dec, optimizer, validation_loader, tensorboard_writer)
        train_results = (losses, validation_data, dices_iou)
        self._save_train_results(train_results)
        self._save_model(model)

        self.eval(model=model, device=device, save_images=self.cfg.SAVE_IMAGES, plot_seg=False, reload_final=False, dice_threshold=dice_threshold)

        # Dodana evalvacija na TRAIN setu
        self.eval(model=model, device=device, save_images=False, plot_seg=False, reload_final=False, dice_threshold=dice_threshold, eval_loader=train_loader)

        self._save_params()

    def eval(self, model, device, save_images, plot_seg, reload_final, dice_threshold, eval_loader=None):
        self.reload_model(model, reload_final)
        is_validation = True
        if eval_loader is None:
            eval_loader = get_dataset("TEST", self.cfg)
            is_validation = False
        self.eval_model(device, model, eval_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=is_validation, plot_seg=plot_seg, dice_threshold=dice_threshold)

    def training_iteration(self, data, device, model, criterion_seg, criterion_seg_upsampled, criterion_dec, optimizer, weight_loss_seg, weight_loss_dec,
                           tensorboard_writer, iter_index):
        images, seg_masks, seg_loss_masks, is_segmented, _, seg_mask_bins, seg_loss_mask_bins = data

        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for >1

        num_subiters = int(batch_size / memory_fit)
        #
        total_loss = 0
        total_correct = 0

        optimizer.zero_grad()

        total_loss_seg = 0
        total_loss_dec = 0

        for sub_iter in range(num_subiters):
            images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_masks_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_loss_masks_ = seg_loss_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_mask_bins_ = seg_mask_bins[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_loss_mask_bins_ = seg_loss_mask_bins[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            is_pos_ = seg_masks_.max().reshape((memory_fit, 1)).to(device)

            if tensorboard_writer is not None and iter_index % 100 == 0:
                tensorboard_writer.add_image(f"{iter_index}/image", images_[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_index}/seg_mask", seg_masks[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_index}/seg_loss_mask", seg_loss_masks_[0, :, :, :])

            decision, output_seg_mask, seg_mask_upsampled = model(images_)

            if is_segmented[sub_iter]:
                if self.cfg.WEIGHTED_SEG_LOSS:
                    loss_seg = torch.mean(criterion_seg(output_seg_mask, seg_masks_) * seg_loss_masks_)
                    loss_seg_upsampling = torch.mean(criterion_seg_upsampled(seg_mask_upsampled, seg_mask_bins_) * seg_loss_mask_bins_) # Loss funkcija za upsamplano segmentacijo (BCE)
                else:
                    loss_seg = criterion_seg(output_seg_mask, seg_masks_)
                    loss_seg_upsampling = criterion_seg_upsampled(seg_mask_upsampled, seg_mask_bins_)                                   # Loss funkcija za upsamplano segmentacijo (BCE)
                loss_dec = criterion_dec(decision, is_pos_)

                # Loss funkcijo za upsamplano segmentacijo prištejem skupnemu lossu za segmentacijo
                total_loss_seg += loss_seg.item() + loss_seg_upsampling.item()
                total_loss_dec += loss_dec.item()

                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_seg * (loss_seg + loss_seg_upsampling) + weight_loss_dec * loss_dec # Dodal loss_seg_upsampling
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

    def _train_model(self, device, model, train_loader, criterion_seg, criterion_seg_upsampled, criterion_dec, optimizer, validation_set, tensorboard_writer):
        losses = []
        validation_data = []
        dices_iou = []
        dice_threshold = 0
        max_validation = -1
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
                                                                                           criterion_seg_upsampled,
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

            self._log(
                f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}, correct={epoch_correct}/{samples_per_epoch}, in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/Train/segmentation", epoch_loss_seg, epoch)
                tensorboard_writer.add_scalar("Loss/Train/classification", epoch_loss_dec, epoch)
                tensorboard_writer.add_scalar("Loss/Train/joined", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Accuracy/Train/", epoch_correct / samples_per_epoch, epoch)

            if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1):
                validation_ap, validation_accuracy, dice_threshold, dice, iou = self.eval_model(device=device, model=model, eval_loader=validation_set, save_folder=None, save_images=False, is_validation=True, plot_seg=False, dice_threshold=None)
                validation_data.append((validation_ap, epoch))
                dices_iou.append((epoch, dice, iou))

                if validation_ap > max_validation:
                    max_validation = validation_ap
                    self._save_model(model, "best_state_dict.pth")

                model.train()
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("Accuracy/Validation/", validation_accuracy, epoch)

        return losses, validation_data, dice_threshold, dices_iou

    def eval_model(self, device, model, eval_loader, save_folder, save_images, is_validation, plot_seg, dice_threshold):
        model.eval()

        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT

        res = []
        predictions, ground_truths = [], []

        images, predicted_segs, true_segs = [], [], []
        predicted_segs_pos, predicted_segs_neg = [], []

        for data_point in eval_loader:
            image, seg_mask, seg_loss_mask, _, sample_name, seg_mask_original, seg_loss_mask_original = data_point
            image, seg_mask = image.to(device), seg_mask.to(device)
            is_pos = (seg_mask.max() > 0).reshape((1, 1)).to(device).item() # Bool - seg_mask pozitivna ali ne
            prediction, pred_seg, pred_seg_upsampled = model(image)
            prediction = nn.Sigmoid()(prediction)
            pred_seg = nn.Sigmoid()(pred_seg)
            pred_seg_upsampled = nn.Sigmoid()(pred_seg_upsampled)

            prediction = prediction.item()
            image = image.detach().cpu().numpy()
            pred_seg = pred_seg.detach().cpu().numpy()
            seg_mask = seg_mask.detach().cpu().numpy()
            pred_seg_upsampled = pred_seg_upsampled.detach().cpu().numpy()

            image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pred_seg = cv2.resize(pred_seg[0, 0, :, :], dsize) if len(pred_seg.shape) == 4 else cv2.resize(pred_seg[0, :, :], dsize)

            predictions.append(prediction)
            ground_truths.append(is_pos)
            res.append((prediction, None, None, is_pos, sample_name[0]))

            pred_seg_upsampled = pred_seg_upsampled[0][0]
            seg_mask_original = seg_mask_original.numpy()[0][0]
            predicted_segs.append(pred_seg_upsampled)
            if is_pos:
                predicted_segs_pos.append(pred_seg_upsampled)
            else:
                predicted_segs_neg.append(pred_seg_upsampled)
            true_segs.append(seg_mask_original)
            images.append(image)

            if not is_validation:
                if save_images:
                    if self.cfg.WEIGHTED_SEG_LOSS:
                        #seg_loss_mask = cv2.resize(seg_loss_mask.numpy()[0, 0, :, :], dsize)
                        utils.plot_sample(sample_name[0], image, pred_seg, seg_loss_mask_original.numpy()[0][0], save_folder, pred_seg_upsampled, decision=prediction, plot_seg=plot_seg)
                    else:
                        utils.plot_sample(sample_name[0], image, pred_seg, seg_mask_original, save_folder, pred_seg_upsampled, decision=prediction, plot_seg=plot_seg)

        if is_validation:
            # Računanje dice thresholda
            # 1. Minimum maksimalnih pikslov vseh predikcij
            if self.cfg.DICE_THRESHOLD == 1:
                max_pixels_pos = np.amax(np.amax(np.array(predicted_segs_pos), axis=1), axis=1) # Max piksli pozitivnih predicted segmentacij
                min_pixel_of_max_pixels_pos = max_pixels_pos.min().item() # Min piksel vseh max pikslov
                
                max_pixels_neg = np.amax(np.amax(np.array(predicted_segs_neg), axis=1), axis=1) # Max piksli negativnih predicted segmentacij
                min_pixel_of_max_pixels_neg = max_pixels_neg.min().item() # Min piksel vseh max pikslov
                
                dice_threshold = min_pixel_of_max_pixels_pos
            
            # 2. precision_recall, subsampling
            elif self.cfg.DICE_THRESHOLD == 2:
                dice_metrics = utils.get_metrics(np.array(true_segs, dtype=bool).flatten()[::self.cfg.DICE_THR_FACTOR], np.array(predicted_segs).flatten()[::self.cfg.DICE_THR_FACTOR]) # vsak 10. piksel GT-jev damo v 1D bool array, vsak 10. piksel predicted segmentacij v 1D float array
                dice_threshold = dice_metrics['best_thr']

            metrics = utils.get_metrics(np.array(ground_truths), np.array(predictions))
            FP, FN, TP, TN = list(map(sum, [metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"]]))
            dice_mean, dice_std, iou_mean, iou_std = utils.dice_iou(predicted_segs, true_segs, dice_threshold)
            self._log(f"VALIDATION on {eval_loader.dataset.kind} set || AUC={metrics['AUC']:f}, and AP={metrics['AP']:f}, with best thr={metrics['best_thr']:f} "
                      f"at f-measure={metrics['best_f_measure']:.3f} and FP={FP:d}, FN={FN:d}, TOTAL SAMPLES={FP + FN + TP + TN:d}\nDice: mean: {dice_mean:f}, std: {dice_std:f}, IOU: mean: {iou_mean:f}, std: {iou_std:f}, Dice Threshold: {dice_threshold:f}")
            
            if self.cfg.DICE_THRESHOLD == 1:
                self._log(f"Min of max pixels: POSITIVE: {min_pixel_of_max_pixels_pos}, NEGATIVE: {min_pixel_of_max_pixels_neg}")

            return metrics["AP"], metrics["accuracy"], dice_threshold, dice_mean, iou_mean
        else:
            utils.evaluate_metrics(samples=res, results_path=self.run_path, run_name=self.run_name, segmentation_predicted=predicted_segs, segmentation_truth=true_segs, images=images, dice_threshold=dice_threshold, dataset_kind=eval_loader.dataset.kind)

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
            model.load_state_dict(torch.load(path))
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
        losses, validation_data, dices_iou = results
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
        epochs, dices, iou = map(list, zip(*dices_iou))
        plt.clf()
        plt.plot(epochs, dices, label="Dice")
        plt.plot(epochs, iou, label="IOU")
        plt.xlabel("Epochs")
        plt.ylabel("Dice")
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "dice_iou"), dpi=200)

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

    def _save_model(self, model, name="final_state_dict.pth"):
        output_name = os.path.join(self.model_path, name)
        self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)

        torch.save(model.state_dict(), output_name)

    def _get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE)

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