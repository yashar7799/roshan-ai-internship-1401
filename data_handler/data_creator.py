import shutil
import numpy as np
import os
from glob import glob
import splitfolders


class DataCreator():

    """
    Labels
    ------
    Restored
    Normal
    Restored_Mild_Caries
    Restored_Deep_Caries
    Deep_Caries
    Mild_Caries
    """

    def __init__(self, src:str, dst:str, n_classes:int = 6):

        self.src = src
        self.dst = dst
        self.n_classes = n_classes

        if n_classes == 6:
            classes = ['Restored', 'Deep_Caries', 'Mild_Caries', 'Normal', 'Restored_Mild_Caries', 'Restored_Deep_Caries']
        else:
            classes = ['normal', 'restored', 'caries']

        self.classes = classes

    def create_data_folder(self):
        
        shutil.rmtree(self.dst, ignore_errors=True)

        shutil.copytree(src=self.src, dst=self.dst)

        if self.n_classes == 3:
            normal_images = []
            restored_images = []
            caries_images = []
            for cls in ['Restored', 'Deep_Caries', 'Mild_Caries', 'Normal', 'Restored_Mild_Caries', 'Restored_Deep_Caries']:
                if cls in ['Restored', 'Restored_Mild_Caries', 'Restored_Deep_Caries']:
                    restored_images.extend([os.path.join(self.dst, cls, im) for im in os.listdir(os.path.join(self.dst, cls))])
                elif cls in ['Deep_Caries', 'Mild_Caries']:
                    caries_images.extend([os.path.join(self.dst, cls, im) for im in os.listdir(os.path.join(self.dst, cls))])
                else:
                    normal_images.extend([os.path.join(self.dst, cls, im) for im in os.listdir(os.path.join(self.dst, cls))])

            os.makedirs(os.path.join(self.dst, 'normal'), exist_ok=True)
            os.makedirs(os.path.join(self.dst, 'restored'), exist_ok=True)
            os.makedirs(os.path.join(self.dst, 'caries'), exist_ok=True)

            for image in normal_images:
                shutil.copy(image, os.path.join(self.dst, 'normal'))
            shutil.rmtree(os.path.join(self.dst, 'Normal'), ignore_errors=True)

            for image in restored_images:
                shutil.copy(image, os.path.join(self.dst, 'restored'))
            shutil.rmtree(os.path.join(self.dst, 'Restored'), ignore_errors=True)
            shutil.rmtree(os.path.join(self.dst, 'Restored_Mild_Caries'), ignore_errors=True)
            shutil.rmtree(os.path.join(self.dst, 'Restored_Deep_Caries'), ignore_errors=True)

            for image in caries_images:
                shutil.copy(image, os.path.join(self.dst, 'caries'))
            shutil.rmtree(os.path.join(self.dst, 'Deep_Caries'), ignore_errors=True)
            shutil.rmtree(os.path.join(self.dst, 'Mild_Caries'), ignore_errors=True)

        print('data folder created successfully.\n')

    def partitioning(self, partitioning_base_folder:str = '../dataset', val_ratio:float = 0.15, test_ratio:float = 0.15, seed:float = None):

        shutil.rmtree(partitioning_base_folder, ignore_errors=True)

        splitfolders.ratio(input=self.dst, output=partitioning_base_folder, ratio=(1-val_ratio-test_ratio, val_ratio, test_ratio), move=False, seed=seed)

        train_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'train')))
        val_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'val')))
        test_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'test')))

        if not train_classes == val_classes == test_classes:
            raise FileNotFoundError('data is not completely ready!\ncheck that you run download_data_folder method correctly.')

        partition = {'train':[], 'val':[], 'test':[]}
        labels = {}

        for cls in self.classes:

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
        
        for cls in self.classes:

            n_train = len(os.listdir(os.path.join(partitioning_base_folder, 'train', cls)))
            n_val = len(os.listdir(os.path.join(partitioning_base_folder, 'val', cls)))
            n_test = len(os.listdir(os.path.join(partitioning_base_folder, 'test', cls)))

            print(f'{cls} >>> train: {n_train} | val: {n_val} | test: {n_test}')

        print('\n')

        return partition, labels