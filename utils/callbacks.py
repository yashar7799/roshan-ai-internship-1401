"""
This module contains a function to provide callbacks.
"""

from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from utils.warmup_onecycle import WarmUpCosineDecayScheduler
from datetime import datetime


def get_callbacks(model_path,
                  early_stopping_patience,
                  tb_log_dir,
                  plateau_reduce_min_lr,
                  plateau_reduce_factor,
                  plateau_reduce_patience,
                  epochs,
                  warmup_epoch,
                  batch_size,
                  warmup_max_lr,
                  sample_count,
                  model_name,
                  n_classes,
                  **kwargs):
    """
    This function use some callbacks from tensorflow.python.keras.callbacks

    Parameters
    ----------
    model_path: str ; path to save the model file.
    early_stopping_p: int ; number of epochs with no improvement after which training will be stopped.
    save_weights_only: bool ; if True, then only the model's weights will be saved.
    plateau_min_lr: float ; lower bound on the learning rate; it is for tensorflow.python.keras.callbacks.ReduceLROnPlateau module
    **kwargs

    Returns
    -------
    checkpoint: a tensorflow.python.keras.callbacks.ModelCheckpoint instance
    reduce_lr: a tensorflow.python.keras.callbacks.ReduceLROnPlateau instance
    early_stopping: a tensorflow.python.keras.callbacks.EarlyStopping instance
    """
    checkpoint = ModelCheckpoint(filepath=model_path, 
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True,
                                 )

    plateau_reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  factor=plateau_reduce_factor,  # new_lr = lr * factor
                                  patience=plateau_reduce_patience,  # number of epochs with no improvment
                                  min_lr=plateau_reduce_min_lr,  # lower bound on the learning rate
                                  mode='min',
                                  verbose=1
                                  )

    total_steps = int(epochs * sample_count / batch_size)
    # Compute the number of warmup batches.
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    # Generate dummy data.
    # warmup_batches = warmup_epoch * sample_count / batch_size

    # Create the Learning rate scheduler.
    warmup_lr = WarmUpCosineDecayScheduler(warmup_max_lr=warmup_max_lr,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0.0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=0, 
                                            verbose=0)

    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_patience, verbose=1)

    tensorboard = TensorBoard(log_dir=tb_log_dir + '/' + model_name + '{}classes__{}'.format(str(n_classes), str(datetime.now()).replace(':', '_').replace(' ','_')),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    return checkpoint, warmup_lr, early_stopping, tensorboard, plateau_reduce_lr
