from config import Config
from end2end import End2End

# Konfiguracija
configuration = Config()

configuration.RUN_NAME = "TEST"
configuration.BATCH_SIZE = 1
configuration.DATASET = "CRACK500"
configuration.DATASET_PATH = "./datasets/CRACK500"
configuration.DELTA_CLS_LOSS = 0.01
configuration.DILATE = 1
configuration.DYN_BALANCED_LOSS = True
configuration.EPOCHS = 5
configuration.FREQUENCY_SAMPLING = False
configuration.GPU = 0
configuration.LEARNING_RATE = 1.0
configuration.WEIGHTED_SEG_LOSS = False
configuration.GRADIENT_ADJUSTMENT = True
configuration.NUM_SEGMENTED = 10 # 6921, 230
configuration.VALIDATE = True
configuration.VALIDATE_ON_TEST = False
configuration.VALIDATION_N_EPOCHS = 5
configuration.USE_BEST_MODEL = True
configuration.HARD_NEG_MINING = 0.2
configuration.BEST_MODEL_TYPE = 'seg'

configuration.init_extra()

# Model

end2end = End2End(cfg=configuration)
end2end.train()