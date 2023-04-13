"""
This is the main module of this project ; here we difine a function to start training process.
"""

import sys
import tensorflow as tf
from datetime import datetime
import os
from models import load_model
# from params import get_args
from params.main import main_args
from data_handler.data_loader import DataGenerator
from data_handler.data_creator import DataCreator
from tensorflow.keras.optimizers import Adam
from utils.callbacks import get_callbacks
from utils.mlflow_handler import MLFlowHandler
from utils.logger import get_logs
from utils.utils import get_gpu_grower

get_gpu_grower()


def train():
    """
    this function is to start trainning process of the desired model.
    it should run in below format:
        python train.py -[option] [value] --[option] [value] ...
    to see available options and arguments see Readme.md file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}\n")
    args = main_args()
    print(f"Arguments: {args}\n")

    id_ = model_name + "_" + str(args.n_classes) + "classes__" + str(datetime.now().date()) + "_" + str(datetime.now().time())
    weights_base_path = os.path.join(args.weights_base_folder, f'{str(args.n_classes)}classes')
    os.makedirs(weights_base_path, exist_ok=True)
    weight_path = os.path.join(weights_base_path, id_) + ".h5"
    
    mlflow_handler = MLFlowHandler(model_name=model_name,
                                   warmup=args.warmup_lr_scheduler,
                                   run_name=id_,
                                   mlflow_source=args.mlflow_source,
                                   run_ngrok=args.run_ngrok,
                                   ngrok_auth_token=args.ngrok_auth_token)
    mlflow_handler.start_run(args)

    data = DataCreator(path_to_images_file=args.path_to_images_file, path_to_metadata_csv_file=args.path_to_metadata_csv_file, dst=args.dataset_dir_before_split)
    partition, labels, encoded_classes_dict = data.partitioning(partitioning_base_folder=args.dataset_dir_after_split, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    unique_classes = list(encoded_classes_dict.keys())
    unique_encoded_classes = list(encoded_classes_dict.values())

    report = list(encoded_classes_dict.items())
    mlflow_handler.add_report(str(report), 'logs/class&index_pairs.txt')

    mlflow_handler.add_report(str(unique_classes), 'logs/classes.txt')

    print(f'classes are: {unique_classes}\n')
    print(f'(class, index) pairs are: {report}\n')

    train_loader = DataGenerator(partition['train'], labels, encoded_classes_dict, batch_size=args.batch_size, dim=(args.input_shape[0], args.input_shape[1]), n_channels=args.input_shape[2], n_classes=args.n_classes, augmentation=args.augmentation, augmentation_prob=args.augmentation_prob)
    val_loader = DataGenerator(partition['val'], labels, encoded_classes_dict, batch_size=args.batch_size, dim=(args.input_shape[0], args.input_shape[1]), n_channels=args.input_shape[2], n_classes=args.n_classes, augmentation=args.augmentation, augmentation_prob=args.augmentation_prob)
    test_loader = DataGenerator(partition['test'], labels, encoded_classes_dict, batch_size=args.batch_size, dim=(args.input_shape[0], args.input_shape[1]), n_channels=args.input_shape[2], n_classes=args.n_classes)
    if args.use_tpu:
        # detect and init the TPU
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

        # instantiate a distribution strategy
        tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

        with tpu_strategy.scope():
            if args.model == 'model1':
                model = load_model(args.model, input_shape=args.input_shape, num_classes=args.n_classes, pre_trained=args.pretrain, model_path=args.path_to_pretrain, dropout=args.dropout_rate)
            elif args.model == 'resnet18':
                model = load_model(args.model, input_shape=args.input_shape, num_classes=args.n_classes, pre_trained=args.pretrain, model_path=args.path_to_pretrain)
            elif args.model in ['resnet50', 'resnet50v2', 'efficientnet_b0']:
                model = load_model(args.model, input_shape=args.input_shape, num_classes=args.n_classes, pre_trained=args.pretrain, model_path=args.path_to_pretrain, imagenet_weights=args.imagenet_weights)
            else:
                model = load_model(args.model, input_shape=args.input_shape, num_classes=args.n_classes, pre_trained=args.pretrain, model_path=args.path_to_pretrain, dropout=args.dropout_rate, imagenet_weights=args.imagenet_weights)

            print("Loading Model is Done!\n")

            if args.pretrain:
                model.load_weights(args.path_to_pretrain)
                print('pretrain weights loaded.\n')
            model.summary()

            checkpoint, warmup_lr, early_stopping, plateau_reduce_lr = get_callbacks(model_path=weight_path,
                                                                                    early_stopping_patience=args.early_stopping_patience,
                                                                                    #    tb_log_dir=args.tb_log_dir,
                                                                                    epochs=args.epochs,
                                                                                    sample_count=len(train_loader) * args.batch_size,
                                                                                    batch_size=args.batch_size,
                                                                                    warmup_epoch=args.warmup_epoch,
                                                                                    warmup_max_lr=args.warmup_max_lr,
                                                                                    #    model_name=model_name,
                                                                                    #    n_classes=args.n_classes,
                                                                                    plateau_reduce_min_lr=args.plateau_reduce_min_lr,
                                                                                    plateau_reduce_factor=args.plateau_reduce_factor,
                                                                                    plateau_reduce_patience=args.plateau_reduce_patience)
            
            callbacks = [checkpoint, early_stopping, mlflow_handler.mlflow_logger]
            
            if args.warmup_lr_scheduler:
                callbacks.append(warmup_lr)
                print('warmup_lr_scheduler activated.\n')
            elif args.cosine_decay_lr_scheduler:
                lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=args.cosine_decay_initial_lr, decay_steps=args.epochs, alpha=args.cosine_decay_alpha, name='None')
                opt = Adam(learning_rate=lr_scheduler)
                print('cosine_decay_lr_scheduler activated.\n')
            elif args.plateau_reduce_lr_scheduler:
                callbacks.append(plateau_reduce_lr)
                opt = Adam(learning_rate=args.plateau_reduce_initial_lr)
                print('plateau_reduce_lr_scheduler activated.\n')
            else:
                opt = Adam(learning_rate=args.learning_rate)
                print(f'none of lr_schedulers  activated, learning rate fixed at: {args.learning_rate}\n')
            
            if args.loss == 'huber':
                loss = tf.keras.losses.Huber()
            else:
                loss = args.loss

            model.compile(optimizer=opt, loss=loss, metrics=['acc'])
        
        model.fit(x=train_loader,
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=val_loader,
                validation_batch_size=args.batch_size,
                callbacks=callbacks,
                use_multiprocessing=args.multiprocessing,
                )
    else:
        if args.model == 'model1':
            model = load_model(args.model, input_shape=args.input_shape, num_classes=args.n_classes, pre_trained=args.pretrain, model_path=args.path_to_pretrain, dropout=args.dropout_rate)
        elif args.model == 'resnet18':
            model = load_model(args.model, input_shape=args.input_shape, num_classes=args.n_classes, pre_trained=args.pretrain, model_path=args.path_to_pretrain)
        elif args.model in ['resnet50', 'resnet50v2', 'efficientnet_b0']:
            model = load_model(args.model, input_shape=args.input_shape, num_classes=args.n_classes, pre_trained=args.pretrain, model_path=args.path_to_pretrain, imagenet_weights=args.imagenet_weights)
        else:
            model = load_model(args.model, input_shape=args.input_shape, num_classes=args.n_classes, pre_trained=args.pretrain, model_path=args.path_to_pretrain, dropout=args.dropout_rate, imagenet_weights=args.imagenet_weights)

        print("Loading Model is Done!\n")

        if args.pretrain:
            model.load_weights(args.path_to_pretrain)
            print('pretrain weights loaded.\n')
        model.summary()

        checkpoint, warmup_lr, early_stopping, plateau_reduce_lr = get_callbacks(model_path=weight_path,
                                                                                early_stopping_patience=args.early_stopping_patience,
                                                                                #    tb_log_dir=args.tb_log_dir,
                                                                                epochs=args.epochs,
                                                                                sample_count=len(train_loader) * args.batch_size,
                                                                                batch_size=args.batch_size,
                                                                                warmup_epoch=args.warmup_epoch,
                                                                                warmup_max_lr=args.warmup_max_lr,
                                                                                #    model_name=model_name,
                                                                                #    n_classes=args.n_classes,
                                                                                plateau_reduce_min_lr=args.plateau_reduce_min_lr,
                                                                                plateau_reduce_factor=args.plateau_reduce_factor,
                                                                                plateau_reduce_patience=args.plateau_reduce_patience)
        
        callbacks = [checkpoint, early_stopping, mlflow_handler.mlflow_logger]
        
        if args.warmup_lr_scheduler:
            callbacks.append(warmup_lr)
            print('warmup_lr_scheduler activated.\n')
        elif args.cosine_decay_lr_scheduler:
            lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args.cosine_decay_initial_lr, decay_steps=args.epochs, alpha=args.cosine_decay_alpha, name='None')
            opt = Adam(learning_rate=lr_scheduler)
            print('cosine_decay_lr_scheduler activated.\n')
        elif args.plateau_reduce_lr_scheduler:
            callbacks.append(plateau_reduce_lr)
            opt = Adam(learning_rate=args.plateau_reduce_initial_lr)
            print('plateau_reduce_lr_scheduler activated.\n')
        else:
            opt = Adam(learning_rate=args.learning_rate)
            print(f'none of lr_schedulers  activated, learning rate fixed at: {args.learning_rate}\n')
        
        if args.loss == 'huber':
            loss = tf.keras.losses.Huber()
        else:
            loss = args.loss

        model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    
        model.fit(x=train_loader,
                batch_size=args.batch_size,
                epochs=args.epochs,
                validation_data=val_loader,
                validation_batch_size=args.batch_size,
                callbacks=callbacks,
                use_multiprocessing=args.multiprocessing,
                workers=args.workers,
                )

    print("Training Model is Done!\n")

    get_logs(model, test_loader, unique_encoded_classes, unique_classes, mlflow_handler)
    mlflow_handler.end_run(weight_path)


if __name__ == '__main__':
    train()