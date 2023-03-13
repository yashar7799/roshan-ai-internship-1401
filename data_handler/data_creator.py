import shutil
import numpy as np
import os
from glob import glob
import splitfolders


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
        
        shutil.rmtree(self.dst.split('.')[0], ignore_errors=True)

        shutil.copy(src=self.src, dst=self.dst)

        shutil.unpack_archive(self.dst, '..')
        print('data folder created successfully.\n')

    def partitioning(self, partitioning_base_folder:str = '../dataset', val_ratio:float = 0.15, test_ratio:float = 0.15, seed:float = None):

        shutil.rmtree(partitioning_base_folder, ignore_errors=True)

        splitfolders.ratio(input=self.dst.split('.')[0], output=partitioning_base_folder, ratio=(1-val_ratio-test_ratio, val_ratio, test_ratio), move=False, seed=seed)

        train_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'train')))
        val_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'val')))
        test_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'test')))

        if not train_classes == val_classes == test_classes:
            raise FileNotFoundError('data is not completely ready!\ncheck that you run create_data_folder method correctly.')

        classes = os.listdir(self.dst.split('.')[0])

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