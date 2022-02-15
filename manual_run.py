from config import Config
from end2end import End2End

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
configuration.GPU = 0
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
end2end.eval(model, device, False, False, False, None)