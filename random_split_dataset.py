import os, random, shutil

random.seed(420)

DATASET = 'SCCDNet_dataset'
FACTOR = 100

path_datasets = './datasets'
path_subset = os.path.join('./datasets', DATASET + '_subset')
dirs = ['test', 'train']

for dir in dirs:
    path_samples = os.path.join('./datasets', DATASET, dir, 'images')
    path_gts = os.path.join('./datasets', DATASET, dir, 'masks')
    new_path_samples = os.path.join(path_subset, dir, 'images')
    new_path_gts = os.path.join(path_subset, dir, 'masks')
    samples = [i for i in sorted(os.listdir(path_samples))]
    number_of_selected_samples = len(samples) // FACTOR
    random_samples = random.sample(samples, number_of_selected_samples)
    if not os.path.exists(new_path_samples):
        os.makedirs(new_path_samples)
    if not os.path.exists(new_path_gts):
        os.makedirs(new_path_gts)
    
    for sample in random_samples:
        shutil.copy2(os.path.join(path_samples, sample), os.path.join(new_path_samples, sample))
        shutil.copy2(os.path.join(path_gts, sample), os.path.join(new_path_gts, sample))