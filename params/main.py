"""
Set main parameters before execution
"""

from argparse import ArgumentParser


def main_args():
    """
    Parameters
    ----------
    None

    Returns
    -------
    Returns some hyper-parameters which are common among all the models.
    """
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='model1', help='model name.', required=True)
    parser.add_argument('--input_shape', type=int, nargs='+', help='desired input shape to feed the model with')
    parser.add_argument('--n_classes', type=int, default=30, help='number of classes; this should be same as the number of classes of the dataset you are using', required=True)
    parser.add_argument('--batch_size', type=int, default=8, help='define batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='define dropout rate to use between fc layers')
    parser.add_argument('--loss', type=str, default='categorical_crossentropy', help='You can pass these losses: categorical_crossentropy | kullback_leibler_divergence | huber')
    parser.add_argument('--epochs', type=int, default=5, help='define number of training epochs')

    parser.add_argument('--in_kaggle', dest='in_kaggle', action='store_true', help="pass this arg if you are running train.py in kaggle")
    parser.add_argument('--not_in_kaggle', dest='in_kaggle', action='store_false', help="pass this arg if you are running train.py everywhere but kaggle")

    parser.add_argument('--val_ratio', type=float, default=0.15, help='validation ratio to be devided from dataset')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='test ratio to be devided from dataset')

    parser.add_argument('--pretrain', dest='pretrain', action='store_true', help='pass this arg if you want to load weights of a pretrained model.')
    parser.add_argument('--path_to_pretrain', type=str, default=None, help='path to a pretrained model weights - .h5 file')
    parser.add_argument('--imagenet_weights', dest='imagenet_weights', action='store_true', help='pass this arg if you want to load weights of the model trained on imagenet dataset.')

    parser.add_argument('--weights_base_folder', type=str, default='./weights', help='this is the base folder that all the weights will be saved there')

    parser.add_argument('--path_to_images_file', type=str, default='/content/drive/MyDrive/roshan_internship_dataset_jpg.zip', help='dataset directory, this directory should contain all images in a source, like google drive folder', required=False)
    parser.add_argument('--path_to_metadata_csv_file', type=str, default='/content/drive/MyDrive/roshan_internship_dataset_jpg.zip', help='dataset directory, this directory should contain all images in a source, like google drive folder', required=False)
    parser.add_argument('--dataset_dir_before_split', type=str, default='/content/roshan_internship_dataset_png', help='dataset directory, this directory should contain all images before splitting', required=True)
    parser.add_argument('--dataset_dir_after_split', type=str, default='../dataset', help='dataset directory, this directory should contain train, val & test folders', required=True)

    parser.add_argument('--mlflow_source', type=str, default='./mlruns', help='The mlflow direcotry')
    parser.add_argument('--run_ngrok', dest='run_ngrok', action='store_true', help="pass this arg if you want to run train.py in colab!")
    parser.add_argument('--no_run_ngrok', dest='run_ngrok', action='store_false', help="pass this arg if you want to run train.py locally!")
    parser.add_argument('--ngrok_auth_token', type=str, help='an authentication token that ngrok gives it to you')

    parser.add_argument('--augmentation', dest='augmentation', action='store_true', help='pass this arg if you want augmentations')
    parser.add_argument('--augmentation_p', type=float, default=0.5, help='augmentation probability')

    parser.add_argument('--multiprocessing', dest='multiprocessing', action='store_true', help="Run model.fit with multi-processing")
    parser.add_argument('--no_multiprocessing', dest='multiprocessing', action='store_false', help="Run model.fit without multi-processing")
    parser.add_argument('--workers', type=int, default=1, help="number of workers for model.fit")

    parser.add_argument('--early_stopping_patience', type=int, default=6, help='early stopping patience')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate, use this if you dont use any lr_scheduler')

    parser.add_argument('--plateau_reduce_lr_scheduler', dest='plateau_reduce_lr_scheduler', action='store_true', help="use plateau_reduce_lr_scheduler")
    parser.add_argument('--plateau_reduce_initial_lr', type=float, default=0.001, help='initilal learning rate for plateau_reduce_lr_scheduler')
    parser.add_argument('--plateau_reduce_factor', type=float, default=0.8, help='factor by which the learning rate will be reduced; new_lr = previous_lr * factor')
    parser.add_argument('--plateau_reduce_min_lr', type=float, default=0.0001, help='lower bound on the learning rate for plateau_reduce_lr_scheduler')
    parser.add_argument('--plateau_reduce_patience', type=int, default=4, help='number of epochs with no improvement after which learning rate will be reduced.')

    parser.add_argument('--warmup_lr_scheduler', dest='warmup_lr_scheduler', action='store_true', help="use warmup_lr_scheduler")
    parser.add_argument('--warmup_max_lr', type=float, default=0.1, help='maximum lr that warmup_lr_scheduler will reach to')
    parser.add_argument('--warmup_epoch', type=int, default=3, help='number of epoch to increase the lr to warmup_max_lr')

    parser.add_argument('--cosine_decay_lr_scheduler', dest='cosine_decay_lr_scheduler', action='store_true', help="use cosine_decay_lr_scheduler")
    parser.add_argument('--cosine_decay_initial_lr', type=float, default=0.1, help='cosine_decay_lr_scheduler initial learning_rate')
    parser.add_argument('--cosine_decay_alpha', type=float, default=0.001, help='minimum learning_rate value as a fraction of cosine_decay_initial_lr.')

    parser.add_argument('--tb_log_dir', type=str, default='./tb_logs', help='The TensorBoard directory')

    parser.set_defaults(pretrain=False, imagenet_weights=False, run_ngrok=False, augmentation=False, multiprocessing=False, warmup_lr_scheduler=False, plateau_reduce_lr_scheduler=False, cosine_decay_lr_scheduler=False)

    return parser.parse_args()