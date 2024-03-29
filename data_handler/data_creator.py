import shutil
import numpy as np
import pandas as pd
import os
from glob import glob
import splitfolders
import tarfile
from sklearn.preprocessing import LabelEncoder



class DataCreator():

    """
    Labels
    ------
    34 labels, different flowers' categories.
    """

    def __init__(self, path_to_images_file:str, path_to_metadata_csv_file:str, dst:str):

        self.path_to_images_file = path_to_images_file
        self.path_to_csv = path_to_metadata_csv_file
        self.dst = dst

    def create_data_folder(self, in_kaggle:bool=False, n_classes:int=30):
        
        shutil.rmtree(self.dst, ignore_errors=True)
        os.makedirs(self.dst, exist_ok=True)

        if in_kaggle:
            os.makedirs(os.path.join(self.dst, self.path_to_images_file.split('/')[-1]), exist_ok=True)
            for image in os.listdir(self.path_to_images_file):
                shutil.copy(src=os.path.join(self.path_to_images_file, image), dst=os.path.join(self.dst, self.path_to_images_file.split('/')[-1]))
        else:
            shutil.copy(src=self.path_to_images_file, dst=self.dst)
            shutil.unpack_archive(os.path.join(self.dst, self.path_to_images_file.split('/')[-1]), self.dst)

        shutil.copy(src=self.path_to_csv, dst=self.dst)

        metadata = pd.read_csv(os.path.join(self.dst, 'ordibehesht.csv'))
        metadata.dropna(inplace=True)
        if in_kaggle:
            image_names = os.listdir(os.path.join(self.dst, self.path_to_images_file.split('/')[-1]))
        else:
            image_names = os.listdir(os.path.join(self.dst, self.path_to_images_file.split('/')[-1].split('.')[0]))
        
        image_names = [image_name.split('.')[0] for image_name in image_names]

        for image_dir, labels_list in metadata.itertuples(index=False, name=None):
            labels_list = labels_list.replace('"', '').replace(']', '').replace('[', '').split(',')
            if 'true' in labels_list:
                label = labels_list[labels_list.index('true') - 1]
            else:
                continue

            image_name = image_dir.split('/')[-1].split('.')[0]
            if image_name in image_names:
                os.makedirs(os.path.join(self.dst, 'dataset', label), exist_ok=True)
                shutil.move(src=os.path.join(self.dst, 'roshan_internship_full_dataset_cleaned', image_name + '.jpg'), dst=os.path.join(self.dst, 'dataset', label))
                image_names.remove(image_name)

            if len(image_names) == 0:
                break
        
        # remove 'بدون گیاه' class:
        shutil.rmtree(os.path.join(self.dst, 'dataset', 'بدون گیاه'))

        all_classes = os.listdir(os.path.join(self.dst, 'dataset'))

        len_of_classes_dict = {}

        for cls in all_classes:
            len_of_class = len(os.listdir(os.path.join(self.dst, 'dataset', cls)))
            len_of_classes_dict[cls] = len_of_class
        
        sorted_len_of_classes_dict = dict(sorted(len_of_classes_dict.items(), key=lambda item: item[1], reverse=True))
        len_of_needed_classes_dict = dict(list(sorted_len_of_classes_dict.items())[:n_classes])

        needed_classes = list(len_of_needed_classes_dict.keys())

        # remove no needed classes:
        no_needed_classes = list(set(all_classes).difference(set(needed_classes)))
        for cls in no_needed_classes:
            shutil.rmtree(os.path.join(self.dst, 'dataset', cls), ignore_errors=True)

        print('data folder created successfully.\n')

    def partitioning(self, partitioning_base_folder:str = '../dataset', val_ratio:float = 0.15, test_ratio:float = 0.15, seed:int = 1337):

        shutil.rmtree(partitioning_base_folder, ignore_errors=True)

        classes = os.listdir(os.path.join(self.dst, 'dataset'))

        le = LabelEncoder()
        le.fit(classes)
        classes = list(le.classes_)
        encoded_classes = list(le.transform(classes))

        encoded_classes_dict = {}
        
        for cls, encoded_cls in zip(classes, encoded_classes):
            encoded_classes_dict[cls] = encoded_cls

        splitfolders.ratio(input=os.path.join(self.dst, 'dataset'), output=partitioning_base_folder, ratio=(1-val_ratio-test_ratio, val_ratio, test_ratio), seed=seed)

        train_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'train')))
        val_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'val')))
        test_classes = sorted(os.listdir(os.path.join(partitioning_base_folder, 'test')))

        if not train_classes == val_classes == test_classes:
            raise FileNotFoundError('data is not completely ready!\ncheck that you run create_data_folder method correctly.')

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

        print(f'Number of classes: {len(classes)}\n')

        # print out train/val/test counts:

        print('Classes and train/val/test counts:\n')
        
        for cls in classes:

            n_train = len(os.listdir(os.path.join(partitioning_base_folder, 'train', cls)))
            n_val = len(os.listdir(os.path.join(partitioning_base_folder, 'val', cls)))
            n_test = len(os.listdir(os.path.join(partitioning_base_folder, 'test', cls)))
            n_all = n_train + n_val + n_test

            print(f'Total number of images from {cls} class: {n_all}')

            print(f'{cls} >>> train: {n_train} | val: {n_val} | test: {n_test}\n')

        print('\n')

        return partition, labels, encoded_classes_dict