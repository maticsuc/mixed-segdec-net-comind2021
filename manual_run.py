from config import Config
from end2end import End2End

# Konfiguracija
configuration = Config()

configuration.RUN_NAME = "TEST"
configuration.BATCH_SIZE = 1
configuration.DATASET = "sccdnet"
configuration.DATASET_PATH = "./datasets/SCCDNet_dataset_subset"
configuration.DELTA_CLS_LOSS = 0.1
configuration.DILATE = 1
configuration.DYN_BALANCED_LOSS = False
configuration.EPOCHS = 5
configuration.FREQUENCY_SAMPLING = False
configuration.GPU = 0
configuration.LEARNING_RATE = 0.001
configuration.WEIGHTED_SEG_LOSS = False
configuration.GRADIENT_ADJUSTMENT = True
configuration.NUM_SEGMENTED = 61
configuration.VALIDATE = True
configuration.VALIDATE_ON_TEST = True
configuration.VALIDATION_N_EPOCHS = 1
configuration.USE_BEST_MODEL = True
#configuration.HARD_NEG_MINING = [5, 0.1, 2]
configuration.BEST_MODEL_TYPE = 'both'
configuration.BCE_LOSS_W = True
configuration.REPRODUCIBLE_RUN = 420
configuration.PXL_DISTANCE = 0
configuration.ON_DEMAND_READ = False

configuration.init_extra()

# Model

end2end = End2End(cfg=configuration)
end2end.train()