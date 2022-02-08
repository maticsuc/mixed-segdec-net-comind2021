from config import Config
from end2end import End2End

# Konfiguracija
configuration = Config()

configuration.GPU = 0
configuration.RUN_NAME = "upsampling_N"
configuration.DATASET = "crack_segmentation"
configuration.DATASET_PATH = "./datasets/crack_segmentation_subset"
configuration.EPOCHS = 10
configuration.LEARNING_RATE = 1.0
configuration.DELTA_CLS_LOSS = 0.01
configuration.BATCH_SIZE = 1
configuration.WEIGHTED_SEG_LOSS = True
configuration.WEIGHTED_SEG_LOSS_P = 2
configuration.WEIGHTED_SEG_LOSS_MAX = 1
configuration.DYN_BALANCED_LOSS = True
configuration.GRADIENT_ADJUSTMENT = True
configuration.FREQUENCY_SAMPLING = True
configuration.NUM_SEGMENTED = 138 # 6921, 230
configuration.VALIDATE = True
configuration.VALIDATE_ON_TEST = False
configuration.VALIDATION_N_EPOCHS = 5
configuration.USE_BEST_MODEL = True
configuration.DICE_THRESHOLD = 1

configuration.init_extra()

# Model

end2end = End2End(cfg=configuration)
end2end.train()
"""
device = end2end._get_device()
model = end2end._get_model().to(device)
end2end.set_dec_gradient_multiplier(model, 0.0)
end2end._set_results_path()

end2end.eval(model, device, True, False, False, 0.5)

"""