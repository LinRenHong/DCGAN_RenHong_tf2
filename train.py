# -*- coding=UTF-8 -*-

import os
import datetime
import tensorflow as tf

from config import config
from utils.datasets import ReadCsvDataSet
from utils.compiler import ModelCompiler

from models.dcgan.dcgan_model import Generator, Discriminator
from utils.loss import generator_loss, discriminator_loss

if __name__ == '__main__':
    opt = config
    IMAGE_FORMAT = 'jpg'

    # Experiment index
    exp_cls = "A"
    exp_idx = 1
    exp_name = "%s%03d" % (exp_cls, exp_idx)

    # Validation index
    val_idx = 2  # Or your can use config to modify: val_idx = opt.val_index

    CSV_PATH = opt.dataset_path + "/" + "tree_and_val_idx.csv"
    IMAGE_CONTAINER = opt.dataset_path

    # Training dataset read from CSV
    train_dataset = ReadCsvDataSet(csv_file_path_=CSV_PATH,
                                   image_container_=IMAGE_CONTAINER,
                                   mode='train',
                                   validation_index_=val_idx).getDataSet()

    # Validation dataset read from CSV
    validation_dataset = ReadCsvDataSet(csv_file_path_=CSV_PATH,
                                        image_container_=IMAGE_CONTAINER,
                                        mode='val',
                                        validation_index_=val_idx).getDataSet()

    print("Training data format: {}".format(train_dataset.take(1).element_spec))
    print("DataSet size (train): {}".format(tf.data.experimental.cardinality(train_dataset).numpy()))
    # print("DataSet size (val): {}".format(tf.data.experimental.cardinality(validation_dataset).numpy()))

    # Get dataset name
    dataset_name = os.path.split(opt.dataset_path)[-1]

    # Get today datetime
    today = datetime.date.today()
    today = "%d%02d%02d" % (today.year, today.month, today.day)

    # Model
    model_name = 'DCGAN'

    # Pre-train model
    # load_model_path = r"pretrain_models/UNet_600.pth" # Or your can use config to modify: load_model_path = opt.load_model_path

    # Checkpoint name
    save_ckpt_name = r"%s-%s-%s-(%s)-ep(%d)-bs(%d)-lr(%s)-img_size(%d, %d, %d)-crop_size(%d, %d, %d)-val_index(%d)" \
                     % (exp_name, today, dataset_name, model_name, opt.n_epochs, opt.batch_size, opt.lr, opt.img_height,
                        opt.img_width, opt.channels, opt.img_crop_height, opt.img_crop_width, opt.channels, val_idx)

    # Training configuration & hyper-parameters
    training_configuration = {
        "validation_index": val_idx,
        "today": today,
        "dataset_name": dataset_name,
        "model": {"generator": Generator(),
                  "discriminator": Discriminator()},
        "model_name": model_name,

        # "load_model_path": load_model_path, # retrain

        "train_dataset": train_dataset.batch(opt.batch_size),
        # "validation_dataset": validation_dataset.batch(opt.val_batch_size),

        "loss_function": {"generator_loss": generator_loss,
                          "discriminator_loss": discriminator_loss},

        "optimizer": {"generator_optimizer": tf.keras.optimizers.Adam(opt.lr,
                                                                      beta_1=opt.b1,
                                                                      beta_2=opt.b2),
                      "discriminator_optimizer": tf.keras.optimizers.Adam(opt.lr, 
                                                                      beta_1=opt.b1,
                                                                      beta_2=opt.b2)},
        # "show_train_result_at_step": 1,  # for terminal log

        "save_ckpt_in_path": save_ckpt_name,
        "tensorboard_path": os.path.join("tf_log", save_ckpt_name),
    }

    
    print("\n############################ Generator: ############################\n")
    generator = Generator()
    noise = tf.random.normal([1, 100])
    fake_image = generator(noise)
    generator.summary()

    print("\n\n########################## Discriminator: ##########################\n")

    discriminator = Discriminator()
    discriminator(fake_image)
    discriminator.summary()

    train_compiler = ModelCompiler(**training_configuration)
    train_compiler.train()
