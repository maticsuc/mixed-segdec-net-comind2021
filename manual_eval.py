from statistics import mode
from config import Config
from end2end import End2End
from data.dataset_catalog import get_dataset

# Konfiguracija
configuration = Config()

configuration.RUN_NAME = "upsampling_8"
configuration.BATCH_SIZE = 1
configuration.DATASET = "crack_segmentation"
configuration.DATASET_PATH = "./datasets/crack_segmentation"
configuration.DELTA_CLS_LOSS = 0.01
configuration.DICE_THRESHOLD = 2
configuration.DICE_THR_FACTOR = 1
configuration.DILATE = 1
configuration.DYN_BALANCED_LOSS = True
configuration.EPOCHS = 50
configuration.FREQUENCY_SAMPLING = True
configuration.GPU = 3
configuration.LEARNING_RATE = 1.0
configuration.WEIGHTED_SEG_LOSS = False
configuration.GRADIENT_ADJUSTMENT = True
configuration.NUM_SEGMENTED = 6921 # 6921, 230
configuration.VALIDATE = True
configuration.VALIDATE_ON_TEST = False
configuration.VALIDATION_N_EPOCHS = 5
configuration.USE_BEST_MODEL = True

configuration.init_extra()

# Model

end2end = End2End(cfg=configuration)
device = end2end._get_device()
model = end2end._get_model().to(device)
end2end.set_dec_gradient_multiplier(model, 0.0)
end2end._set_results_path()
end2end.reload_model(model=model, load_final=False)

# Validacija na VAL setu

validation_loader = get_dataset("VAL", end2end.cfg)
_, _, val_metrics = end2end.eval_model(device=device, model=model, eval_loader=validation_loader, save_folder=end2end.outputs_path, save_images=False, is_validation=True, plot_seg=False, dice_threshold=None)
end2end._log(f"From evaluation on VAL set. Dice threshold: {val_metrics['dice_threshold']:f}, Decision threshold: {val_metrics['best_threshold_dec']:f}")

# Evalvacija na TEST setu

test_loader = get_dataset("TEST", end2end.cfg)
end2end.eval_model(device=device, model=model, eval_loader=test_loader, save_folder=end2end.outputs_path, save_images=end2end.cfg.SAVE_IMAGES, is_validation=False, plot_seg=False, dice_threshold=val_metrics['dice_threshold'], dec_threshold=val_metrics['best_threshold_dec'], two_pxl_threshold=val_metrics['two_pxl_threshold'])
