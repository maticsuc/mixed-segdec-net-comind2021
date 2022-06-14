from config import Config
from end2end import End2End
from data.dataset_catalog import get_dataset
import sys, os

# nohup python -u manual_eval.py 5 crack500_3_1_van CRACK500 > eval_1.out 2>&1 &

#gpu, run_name, dataset = sys.argv[1:]
gpu, run_name, dataset = 0, "crack500_3_1_van", "CRACK500"

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

configuration.init_extra()

# Model

end2end = End2End(cfg=configuration)
end2end._set_results_path()
device = end2end._get_device()
model = end2end._get_model().to(device)
end2end.set_dec_gradient_multiplier(model, 0.0)
end2end.reload_model(model=model, load_final=False)

# Validacija na VAL setu

validation_loader = get_dataset("VAL", end2end.cfg)
_, _, val_metrics = end2end.eval_model(device=device, model=model, eval_loader=validation_loader, save_folder=end2end.outputs_path, save_images=False, is_validation=True, plot_seg=False)
end2end._log(f"From evaluation on VAL set. Decision threshold: {val_metrics['dec_threshold']:f}")
end2end._log(f"From evaluation on VAL set. Segmentation threshold: {val_metrics['two_pxl_threshold']:f}")

# Evalvacija na TEST setu

os.rename(os.path.join(end2end.run_path, 'seg_metrics'), os.path.join(end2end.run_path, 'seg_metrics_test'))

#test_loader = get_dataset("TEST", end2end.cfg)
end2end.eval_model(device=device, model=model, eval_loader=validation_loader, save_folder=end2end.outputs_path, save_images=end2end.cfg.SAVE_IMAGES, is_validation=False, plot_seg=False, dec_threshold=val_metrics['dec_threshold'], two_pxl_threshold=val_metrics['two_pxl_threshold'], dice_threshold=val_metrics['dice_threshold'])
