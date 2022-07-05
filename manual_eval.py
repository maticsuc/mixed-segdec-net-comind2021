from config import Config
from end2end import End2End
from data.dataset_catalog import get_dataset
from utils import create_folder
import sys, os, torch

# nohup python -u manual_eval.py 5 sccdnet_1 sccdnet ./datasets/SCCDNet_dataset sccdnet_1_best_seg > sccdnet_1_best_seg.out 2>&1 &

gpu, run_name, dataset, dataset_path, eval_name = sys.argv[1:]

# Konfiguracija
configuration = Config()

params = [i.replace('\n', '') for i in open(os.path.join('RESULTS', dataset, run_name, 'run_params.txt'), 'r')]

for p in params:
    p, v = p.split(":")
    try:
        v = int(v)
    except:
        try:
            v = float(v)
        except:
            pass

    if v == 'True':
        v = True
    elif v == 'False':
        v = False
    elif v == "None":
        v = None

    setattr(configuration, p, v)

configuration.RUN_NAME = run_name
configuration.GPU = gpu
configuration.DATASET_PATH = dataset_path
configuration.SAVE_IMAGES = True

configuration.init_extra()

# Model

end2end = End2End(cfg=configuration)
end2end._set_results_path()
end2end.set_seed()
device = end2end._get_device()
model = end2end._get_model().to(device)
end2end.set_dec_gradient_multiplier(model, 0.0)
#end2end.reload_model(model=model, load_final=False)

"""
"""
path = os.path.join(end2end.model_path, "ep_175.pth")
model.load_state_dict(torch.load(path, map_location=f"cuda:{end2end.cfg.GPU}"))
end2end._log(f"Loading model state from {path}")

# Make new eval save folder 

end2end.run_path = os.path.join(end2end.cfg.RESULTS_PATH, end2end.cfg.DATASET, eval_name)
end2end.outputs_path = os.path.join(end2end.run_path, "test_outputs")
create_folder(end2end.run_path)
create_folder(end2end.outputs_path)

end2end._log(f"Dataset: {dataset}, Path: {dataset_path}")

# Validacija na VAL setu

validation_loader = get_dataset("VAL", end2end.cfg)
_, _, val_metrics = end2end.eval_model(device=device, model=model, eval_loader=validation_loader, save_folder=end2end.outputs_path, save_images=False, is_validation=True, plot_seg=False)

# Evalvacija na TEST setu

test_loader = get_dataset("TEST", end2end.cfg)
end2end.eval_model(device=device, model=model, eval_loader=test_loader, save_folder=end2end.outputs_path, save_images=end2end.cfg.SAVE_IMAGES, is_validation=False, plot_seg=False, thresholds=val_metrics)