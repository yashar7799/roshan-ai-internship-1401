import shutil
import numpy as np
import pandas as pd
import os
from glob import glob
import splitfolders
import tarfile



class DataCreator():

    """
    Labels
    ------
    34 labels, different flowers' categories.
    """

    def __init__(self, src:str, dst:str):

        self.src = src
        self.dst = dst

    def create_data_folder(self):
        
        shutil.rmtree(self.dst, ignore_errors=True)
        os.makedirs(self.dst, exist_ok=True)

        shutil.copy(src=os.path.join(self.src, 'ordibehesht_10K.tar.gz'), dst=self.dst)
        shutil.copy(src=os.path.join(self.src, 'ordibehesht.csv.tar.gz'), dst=self.dst)

        data_file = tarfile.open(os.path.join(self.dst, 'ordibehesht_10K.tar.gz'))
        data_file.extractall(self.dst)
        data_file.close()

        metadata_file = tarfile.open(os.path.join(self.dst, 'ordibehesht.csv.tar.gz'))
        metadata_file.extractall(self.dst)
        metadata_file.close()

        metadata = pd.read_csv('/content/ordibehesht.csv')
        metadata.dropna(inplace=True)

        image_names = os.listdir(os.path.join(self.dst, 'ordibehesht_images_10000'))

        for image_dir, labels_list in metadata.itertuples(index=False, name=None):
            labels_list = labels_list.replace('"', '').replace(']', '').replace('[', '').split(',')
            if 'true' in labels_list:
                label = labels_list[labels_list.index('true') - 1]

            image_name = image_dir.split('/')[-1]
            if image_name in image_names:
                os.makedirs(os.path.join(self.dst, 'dataset', label), exist_ok=True)
                shutil.move(src=os.path.join(self.dst, 'ordibehesht_images_10000', image_name), dst=os.path.join(self.dst, 'dataset', label))
                image_names.remove(image_name)

            if len(image_names) == 0:
                break

        print('data folder created successfully.\n')

    def partitioning(self, partitioning_base_folder:str = '../dataset', val_ratio:float = 0.15, test_ratio:float = 0.15, seed:float = None):

        shutil.rmtree(partitioning_base_folder, ignore_errors=True)

        splitfolders.ratio(input=os.path.join(self.dst, 'dataset'), output=partitioning_base_folder, ratio=(1-val_ratio-test_ratio, val_ratio, test_ratio), move=False, seed=seed)

        train_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'train')))
        val_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'val')))
        test_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'test')))

        if not train_classes == val_classes == test_classes:
            raise FileNotFoundError('data is not completely ready!\ncheck that you run create_data_folder method correctly.')

        classes = os.listdir(os.path.join(self.dst, 'dataset'))

        partition = {'train':[], 'val':[], 'test':[]}
        labels = {}

        for cls in classes:

            train_files = np.array(glob(os.path.join(partitioning_base_folder, 'train', cls, '*')))
            val_files = np.array(glob(os.path.join(partitioning_base_folder, 'val', cls, '*')))
            test_files = np.array(glob(os.path.join(partitioning_base_folder, 'test', cls, '*')))

            for train in train_files:
                partition['train'].append(train)
                labels[train] = cls

            for val in val_files:
                partition['val'].append(val)
                labels[val] = cls

            for test in test_files:
                partition['test'].append(test)
                labels[test] = cls


        # print out train/val/test counts:

        print('Classes and train/val/test counts:\n')
        
        for cls in classes:

            n_train = len(os.listdir(os.path.join(partitioning_base_folder, 'train', cls)))
            n_val = len(os.listdir(os.path.join(partitioning_base_folder, 'val', cls)))
            n_test = len(os.listdir(os.path.join(partitioning_base_folder, 'test', cls)))

            print(f'{cls} >>> train: {n_train} | val: {n_val} | test: {n_test}')

        print('\n')

        return partition, labels