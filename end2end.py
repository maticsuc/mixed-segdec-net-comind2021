import sched
from tabnanny import verbose
import matplotlib
from sklearn import metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import SegDecNet, FocalLoss, DiceLoss
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
        if self.cfg.REPRODUCIBLE_RUN is not None:
            self._log(f"Reproducible run, fixing all seeds to: {self.cfg.REPRODUCIBLE_RUN}", LVL_DEBUG)
            np.random.seed(self.cfg.REPRODUCIBLE_RUN)
            torch.manual_seed(self.cfg.REPRODUCIBLE_RUN)
            random.seed(self.cfg.REPRODUCIBLE_RUN)
            torch.cuda.manual_seed(self.cfg.REPRODUCIBLE_RUN)
            torch.cuda.manual_seed_all(self.cfg.REPRODUCIBLE_RUN)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = self._get_device()
        model = self._get_model().to(device)
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        loss_seg = self._get_loss(True)

        train_loader = get_dataset("TRAIN", self.cfg)
        validation_loader = get_dataset("VAL", self.cfg)

        tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path) if WRITE_TENSORBOARD else None

        # Save current learning method to model's directory
        utils.save_current_learning_method(save_path=self.run_path)

        losses, validation_data, best_model_metrics, validation_metrics, lrs, difficulty_score_dict = self._train_model(device, model, train_loader, loss_seg, optimizer, scheduler, validation_loader, tensorboard_writer)
        train_results = (losses, validation_data, validation_metrics, lrs)
        self._save_train_results(train_results)
        self._save_model(model)

        # Save difficulty_score_dict
        np.save(os.path.join(self.run_path, "difficulty_score_dict.npy"), difficulty_score_dict)

        self.eval(model=model, device=device, save_images=self.cfg.SAVE_IMAGES, plot_seg=False, reload_final=False, best_model_metrics=best_model_metrics)

        self._save_params()

    def eval(self, model, device, save_images, plot_seg, reload_final, eval_loader=None, best_model_metrics=None):
        self.reload_model(model, reload_final)
        is_validation = True
        if eval_loader is None:
            eval_loader = get_dataset("TEST", self.cfg)
            is_validation = False
        self.eval_model(device, model, eval_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=is_validation, plot_seg=plot_seg, dec_threshold=None, two_pxl_threshold=best_model_metrics['two_pxl_threshold'])

    def training_iteration(self, data, device, model, criterion_seg, optimizer, tensorboard_writer, iter_index):
        images, seg_masks, is_segmented, _, _, _ = data

        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for >1

        num_subiters = int(batch_size / memory_fit)

        optimizer.zero_grad()

        total_loss_seg = 0

        difficulty_score = np.zeros(batch_size)

        for sub_iter in range(num_subiters):
            images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_mask_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)

            if tensorboard_writer is not None and iter_index % 100 == 0:
                tensorboard_writer.add_image(f"{iter_index}/image", images_[0, :, :, :])

            seg_mask_predicted = model(images_)

            if is_segmented[sub_iter]:
                loss_seg = criterion_seg(seg_mask_predicted, seg_mask_)

                if self.cfg.HARD_NEG_MINING is not None:
                    _, _, difficulty_score_mode = self.cfg.HARD_NEG_MINING
                    if difficulty_score_mode == 1:
                        difficulty_score[sub_iter] = loss_seg.item()
                    elif difficulty_score_mode == 2:
                        threshold = 0.5
                        y_true = seg_mask_.detach().cpu().numpy()[0][0].astype(np.uint8)
                        y_pred = (seg_mask_predicted.detach().cpu().numpy()[0][0]>threshold).astype(np.uint8)

                        fp = sum(sum((y_true==0)&(y_pred==1))).item()
                        fn = sum(sum((y_true==1)&(y_pred==0))).item()

                        difficulty_score[sub_iter] = loss_seg.item() * ((2 * fp) + fn + 1)

                total_loss_seg += loss_seg.item()

            loss_seg.backward()

        # Backward and optimize
        optimizer.step()
        optimizer.zero_grad()

        return total_loss_seg, difficulty_score

    def _train_model(self, device, model, train_loader, criterion_seg, optimizer, scheduler, validation_set, tensorboard_writer):
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

        difficulty_score_dict = dict()

        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                self._save_model(model, f"ep_{epoch:02}.pth")

            model.train()

            epoch_loss_seg = 0

            difficulty_score_dict[epoch] = []

            from timeit import default_timer as timer

            time_acc = 0
            start = timer()
            for iter_index, (data) in enumerate(train_loader):
                start_1 = timer()
                curr_loss_seg, difficulty_score = self.training_iteration(data, device, model, criterion_seg, optimizer, tensorboard_writer, (epoch * samples_per_epoch + iter_index))

                end_1 = timer()
                time_acc = time_acc + (end_1 - start_1)

                epoch_loss_seg += curr_loss_seg

                if self.cfg.HARD_NEG_MINING is not None:
                    train_loader.batch_sampler.update_sample_loss_batch(data, difficulty_score, index_key=5)

                difficulty_score_dict[epoch].append({index.item(): round(score, 2) for index, score in zip(data[-1], difficulty_score)})

            end = timer()

            epoch_loss_seg = epoch_loss_seg / samples_per_epoch
            losses.append((epoch_loss_seg, epoch))

            self._log(f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

            if self.cfg.SCHEDULER is not None:
                scheduler.step()
                last_learning_rate = scheduler.get_last_lr()[-1]
                self._log(f"Last computing learning rate by scheduler: {last_learning_rate}")
                lrs.append((epoch, last_learning_rate))
            else:
                lrs.append((epoch, self._get_learning_rate(optimizer=optimizer)))

            self._log(f"Last computing learning rate by optimizer: {self._get_learning_rate(optimizer=optimizer)}")

            if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1):
                validation_ap, validation_accuracy, val_metrics = self.eval_model(device=device, model=model, eval_loader=validation_set, save_folder=None, save_images=False, is_validation=True, plot_seg=False)
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

        return losses, validation_data, best_model_metrics, validation_metrics, lrs, difficulty_score_dict

    def eval_model(self, device, model, eval_loader, save_folder, save_images, is_validation, plot_seg, dec_threshold=None, two_pxl_threshold=None, faktor=None):
        model.eval()

        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT

        images, predicted_segs, true_segs = [], [], []
        samples = {"images": list(), "image_names": list()}

        for data_point in eval_loader:
            image, seg_mask, _, sample_name, _, _ = data_point
            image, seg_mask = image.to(device), seg_mask.to(device)
            seg_mask_predicted = model(image)
            seg_mask_predicted = nn.Sigmoid()(seg_mask_predicted)

            image = image.detach().cpu().numpy()
            seg_mask = seg_mask.detach().cpu().numpy()
            seg_mask_predicted = seg_mask_predicted.detach().cpu().numpy()

            image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
                utils.plot_sample(sample_name[0], image, seg_mask_predicted, seg_mask, save_folder, decision=None, plot_seg=plot_seg)
                utils.save_predicted_segmentation(seg_mask_predicted, sample_name[0], self.run_path)

        if is_validation:
            # Najboljši F1, Pr, Re, threshold
            val_metrics = self.seg_val_metrics(true_segs, predicted_segs, eval_loader.dataset.kind, pxl_distance=self.cfg.PXL_DISTANCE)

            return None, None, val_metrics
        else:
            # Segmentation metrics + vizualizacija
            self._log(f"Evaluation metrics on {eval_loader.dataset.kind} set. {self.cfg.PXL_DISTANCE} pixel distance used.")
           
            pr, re, f1 = utils.segmentation_metrics(seg_truth=true_segs, seg_predicted=predicted_segs, two_pixel_threshold=two_pxl_threshold, samples=samples, run_path=self.run_path, pxl_distance=self.cfg.PXL_DISTANCE)

            self._log(f"Pr: {pr:f}, Re: {re:f}, F1: {f1:f}, threshold: {two_pxl_threshold}")

    def get_dec_gradient_multiplier(self):
        if self.cfg.GRADIENT_ADJUSTMENT:
            grad_m = 0
        else:
            grad_m = 1

        self._log(f"Returning dec_gradient_multiplier {grad_m}", LVL_DEBUG)
        return grad_m

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
        ls, le = map(list, zip(*losses))
        plt.plot(le, ls, label="Loss seg")
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

        df_loss = pd.DataFrame(data={"loss_seg": ls, "epoch": le})
        df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

        if self.cfg.VALIDATE:
            df_loss = pd.DataFrame(data={"validation_data": ls, "loss_seg": ls, "epoch": le})
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
        
        # Loss Segmentation Plot
        plt.clf()
        plt.plot(le, ls)
        plt.xlabel("Epochs")
        plt.ylabel("Loss Segmentation")
        plt.savefig(os.path.join(self.run_path, "loss_seg"), dpi=200)

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
        if self.cfg.SCHEDULER is None:
            return None
        else:
            self._log(f"Using Learning Rate Scheduler: StepLR, Step size: {int(self.cfg.SCHEDULER[0])}, Gamma: {self.cfg.SCHEDULER[1]}")
            return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(self.cfg.SCHEDULER[0]), gamma=self.cfg.SCHEDULER[1])

    def _get_learning_rate(self, optimizer):
        for p in optimizer.param_groups:
            return p["lr"]

    def _get_loss(self, is_seg):
        if is_seg and self.cfg.LOSS == 'focal':
            return FocalLoss().to(self._get_device())
        elif is_seg and self.cfg.LOSS == 'dice':
            return DiceLoss().to(self._get_device())
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