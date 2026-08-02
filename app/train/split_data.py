import os
import shutil
import random
from pathlib import Path

def make_folders(full_dir = "app/data/full_data", data_dir = "app/data"):
    full_path = Path(full_dir)
    data_path = Path(data_dir)
    
    for folder in ['train', 'val', 'test', 'uploaded_train']:
        (data_path / folder).mkdir(parents=True, exist_ok=True)
    
    for classes in os.listdir(full_path):
        class_path = full_path / classes
        if not class_path.is_dir():
            continue
        for split in ['train', 'val', 'test', 'uploaded_train']:
            split_class_path = data_path / split / classes
            split_class_path.mkdir(parents=True, exist_ok=True)
        
def split_data(full_dir = "app/data/full_data", data_dir = "app/data", 
               train = 0.7, val = 0.15, test = 0.15):

    full_path = Path(full_dir)
    data_path = Path(data_dir)
    
    for class_name in os.listdir(full_path):
        class_path = full_path / class_name
        
        if not class_path.is_dir():
            continue
        
        images = list(class_path.glob('*.*'))
        if not images:
            continue
        
        random.shuffle(images)
        
        n = len(images)
        train_end = int(n * train)
        val_end = int(n * (train + val))

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        for split, img_list in [('train', train_images), ('val', val_images), ('test', test_images)]:
            split_class_path = data_path / split / class_name
            split_class_path.mkdir(parents=True, exist_ok=True)
            for img in img_list:
                shutil.copy2(img, split_class_path / img.name)

if __name__ == '__main__':
    make_folders()
    split_data()
        
        
