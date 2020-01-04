# -*- coding=UTF-8 -*-

import os
import datetime
import tensorflow as tf

from config import config
from utils.datasets import ReadCsvPairDataSet
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
    val_idx = 1  # Or your can use config to modify: val_idx = opt.val_index

    CSV_PATH = opt.dataset_path + "/" + "list.csv"
    IMAGE_CONTAINER = opt.dataset_path

    # Training dataset read from CSV
    train_dataset = ReadCsvPairDataSet(csv_file_path_=CSV_PATH,
                                       image_container_=IMAGE_CONTAINER,
                                       mode='train',
                                       validation_index_=val_idx).getDataSet()

    # Validation dataset read from CSV
    validation_dataset = ReadCsvPairDataSet(csv_file_path_=CSV_PATH,
                                            image_container_=IMAGE_CONTAINER,
                                            mode='val',
                                            validation_index_=val_idx).getDataSet()

    print("Training data format: {}".format(train_dataset.take(-1).element_spec))
    print("DataSet size (train): {}".format(tf.data.experimental.cardinality(train_dataset).numpy()))
    print("DataSet size (val): {}".format(tf.data.experimental.cardinality(validation_dataset).numpy()))

    # Get dataset name
    dataset_name = os.path.split(opt.dataset_path)[-1]

    # Get today datetime
    today = datetime.date.today()
    today = "%d%02d%02d" % (today.year, today.month, today.day)

    # Model
    # model = UNet(3, 1)  # UNet(input_channels, output_channels)
    # model_name = model.__class__.__name__
    model_name = 'DCGAN'

    # Pre-train model
    load_model_path = r"pretrain_models/generator_20.h5" # Or your can use config to modify: load_model_path = opt.load_model_path

    # Checkpoint name
    save_ckpt_name = r"%s-%s-%s-(%s)-ep(%d)-bs(%d)-lr(%s)-img_size(%d, %d, %d)-crop_size(%d, %d, %d)-val_index(%d)" \
                     % (exp_name, today, dataset_name, model_name, opt.n_epochs, opt.batch_size, opt.lr, opt.img_height,
                        opt.img_width, opt.channels, opt.img_crop_height, opt.img_crop_width, opt.channels, val_idx)

    # Training configuration & hyper-parameters
    training_configuration = {
        "validation_index": val_idx,
        "today": today,
        "dataset_name": dataset_name,
        # "model": model,
        "model": {"generator": Generator(),
                  "discriminator": Discriminator()},
        "model_name": model_name,

        "load_model_path": load_model_path, # retrain

        "validation_dataset": validation_dataset.batch(opt.val_batch_size),

        "loss_function": {"generator_loss": generator_loss,
                          "discriminator_loss": discriminator_loss},

        "optimizer": {"generator_optimizer": tf.keras.optimizers.Adam(opt.lr),
                      "discriminator_optimizer": tf.keras.optimizers.Adam(opt.lr)},
    }

    # # Show the training set
    # for img, mask in train_dataset.take(5):
    #     print("Img: {}, Mask: {}".format(type(img), mask.shape))
    #     plotImage(img)
    #     # display([img, mask])
    #
    # train_dataset = train_dataset.batch(32)


    validation_compiler = ModelCompiler(**training_configuration)

    for data in train_dataset.take(5):
        print("Img: {}, Mask: {}".format(type(data), data['B'].shape))
        # plotImage(data['A'])
        # display([data['A'], data['B']])

    # model = UpSampling(128, (5, 5), (1, 1))

    print("\n############################ Generator: ############################\n")
    generator = Generator()
    noise = tf.random.normal([1, 100])
    fake_image = generator(noise)
    generator.summary()

    print("\n\n########################## Discriminator: ##########################\n")

    discriminator = Discriminator()
    discriminator(fake_image)
    # discriminator.build((1, 64, 64, 3))
    discriminator.summary()

    validation_compiler.test()
