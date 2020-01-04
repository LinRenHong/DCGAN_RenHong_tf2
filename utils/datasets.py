
import os
import tensorflow as tf
import numpy as np
import pandas as pd

from config import config


opt = config

class ReadCsvDataSet(object):
    def __init__(self, csv_file_path_, image_container_, image_format_=None, mode='train', validation_index_=None):

        self.csv_file_path = csv_file_path_
        self.image_container = image_container_
        self.image_format = image_format_
        self.condition = mode
        self.validation_index = validation_index_

        self.labels = None
        self.fold_index = None
        self.label_names = None
        self.select_indices = None

        self.IMG_HEIGHT = opt.img_height
        self.IMG_WIDTH = opt.img_width
        self.CROP_HEIGHT = opt.img_crop_height
        self.CROP_WIDTH = opt.img_crop_width

        self.df = self._read_csv()

        self.image_paths = self.df["imageFileName"]
        self._check_img()

        self.fold_index = self.df["foldIndex"]

        # self._set_select_indices_by_validation_index()
        # self._set_data_by_validation_index()


    def getDataSet(self):

        return_dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        return_dataset = return_dataset.shuffle(tf.data.experimental.cardinality(return_dataset).numpy())

        ### train ###
        if self.condition.strip() in ["train", "Train", "TRAIN"]:
            return_dataset = return_dataset.map(self.load_image_train,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ### validation ###
        else:
            return_dataset = return_dataset.map(self.load_image_test,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return return_dataset


    def load(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image, tf.float32)

        return image

    def resize(self, image, height, width):
        image = tf.image.resize(image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image

    def random_crop(self, image, height, width):
        cropped_image = tf.image.random_crop(image,
                                             size=[height, width, 3])

        return cropped_image

    def normalize(self, image):
        image = (image / 127.5) - 1

        return image

    @tf.function
    def random_jitter(self, image):
        # resizing to IMG_HEIGHT x IMG_WIDTH x 3
        image = self.resize(image, self.IMG_HEIGHT, self.IMG_WIDTH)

        # randomly cropping to CROP_HEIGHT x CROP_WIDTH x 3
        image = self.random_crop(image, self.CROP_HEIGHT, self.CROP_WIDTH)

        if tf.random.uniform(()) > 0.5:
            # random mirroring
            image = tf.image.flip_left_right(image)

        return image

    def load_image_train(self, image_file):
        image = self.load(image_file)
        image = self.random_jitter(image)
        image = self.normalize(image)

        return {"Image": image}
        # return image, mask

    def load_image_test(self, image_file):
        image = self.load(image_file)
        image = self.resize(image, self.IMG_HEIGHT, self.IMG_WIDTH)
        image= self.normalize(image)

        return image

    def _read_csv(self):
        try:

            df = pd.read_csv(self.csv_file_path)

            return df

        except FileNotFoundError as e:
            print(e, "\nPlease check the csv filepath!")


    def _check_img(self):
        '''
        Check whether images exist in folder or not
        Note: all images should be in image container
        '''

        # check whether image format is already embedded in image paths (check ".")
        if self.image_format is None and "." not in self.image_paths[0]:

            raise FileNotFoundError("There no image format specified in image path {}. "
                                    "Please specify image format".format(self.image_paths[0]))
        elif self.image_format is not None and "." not in self.image_paths[0]:

            print("Concatenate image format into image paths ...")

            self.image_paths = np.array([os.path.join(self.image_container,
                                                      image_path+"."+self.image_format)
                                         for image_path in self.image_paths])
        else:
            self.image_paths = np.array([os.path.join(self.image_container,
                                                      image_path) for image_path in self.image_paths])

        for path in self.image_paths:

            if not os.path.isfile(path):

                raise FileNotFoundError("Image file - {} not found! Please check!".format(path))

        print("All images exist in folder A({}) - {}".format(self.condition, self.image_container))


    def _set_select_indices_by_validation_index(self):
        '''
        Select images by validation index and fold index depending on condition (Train / Validation / Test)
        :return:
        '''

        if self.validation_index is not None and self.fold_index is not None:

            if self.condition.strip() in ["train", "Train", "TRAIN"]:

                indices = self.fold_index != self.validation_index

            else:

                indices = self.fold_index == self.validation_index

            self.select_indices = indices

    def _set_data_by_validation_index(self):
        '''
        Select images by validation index and fold index depending on condition (Train / Validation / Test)
        :return:
        '''

        if self.select_indices is not None:
            self.image_paths = self.image_paths[self.select_indices]
            self.labels = self.labels[self.select_indices] if self.labels is not None else None
            self.label_names = self.label_names[self.select_indices] if self.label_names is not None else None



if __name__ == "__main__":
    CSV_PATH = opt.dataset_path + "/" + "list.csv"
    IMAGE_CONTAINER = opt.dataset_path

    train_dataset = ReadCsvPairDataSet(CSV_PATH, IMAGE_CONTAINER, mode='train', validation_index_=2)
    train_dataset = train_dataset.getDataSet()
    print(train_dataset.take(-1).element_spec)
    print(tf.data.experimental.cardinality(train_dataset).numpy())

    # # Show the training set
    # for img, mask in train_dataset.take(5):
    #     print("Img: {}, Mask: {}".format(type(img), mask.shape))
    #     plotImage(img)
    #     # display([img, mask])
    #
    # train_dataset = train_dataset.batch(32)

    for data in train_dataset.take(5):
        print("Img: {}, Mask: {}".format(type(data), data['B'].shape))
        # display([data['A'], data['B']])
