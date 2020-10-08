
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from time import time
from datetime import datetime
from tqdm import tqdm

from config import config


opt = config

# printout template for log (during training, validation & testing)
log_template = {
        "epoch": "%s: ep %d",
        "step": "step %d",
        "loss": "%s: %.3f",
        "accuracy": "%s: %.4f",
        "step-duration": "%.1f samples/sec; %.2f sec/batch",
        "epoch-duration": "%.2f sec/epoch",
        "time-remaining": "Remaining time: %d hr %d min",
}


class ModelCompiler(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.today = kwargs.get("today", None)
        self.dataset_name = kwargs.get("dataset_name", None)
        self.model_name = kwargs.get("model_name", None)
        self.val_idx = kwargs.get("validation_index", None)
        self.save_ckpt_name = kwargs.get("save_ckpt_in_path", None)
        self.tb_log_path = kwargs.get("tensorboard_path", None)

        # Training dataloader
        self.train_dataset = kwargs.get("train_dataset", None)
        if self.train_dataset is not None:
            self.train_dataset_size = tf.data.experimental.cardinality(self.train_dataset).numpy()

        # Validation dataloader
        self.val_dataset = kwargs.get("validation_dataset", None)
        if self.val_dataset is not None:
            self.val_dataset_size = tf.data.experimental.cardinality(self.val_dataset).numpy()

        # Model
        models = kwargs.get("model", None)
        self.generator = models["generator"]
        self.discriminator = models["discriminator"]

        # Pre-train model
        self.load_model_path = kwargs.get("load_model_path", None)

        # Loss function
        losses = kwargs.get("loss_function", None)
        self.generator_loss = losses["generator_loss"]
        self.discriminator_loss = losses["discriminator_loss"]

        # Optimizer
        optimizers = kwargs.get("optimizer", None)
        self.generator_optimizer = optimizers["generator_optimizer"]
        self.discriminator_optimizer = optimizers["discriminator_optimizer"]
        # Console
        self.show_train_result_at_step = kwargs.get("show_train_result_at_step", None)

        # Seed
        self.seed = tf.random.normal([opt.num_examples_to_generate, opt.noise_dim])


        self.results_dir = "results"
        self.save_images_dir = os.path.join(self.results_dir, "images")
        self.save_models_dir = os.path.join(self.results_dir, "saved_models")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.save_images_dir, exist_ok=True)
        os.makedirs(self.save_models_dir, exist_ok=True)


    def train(self):

        os.makedirs(os.path.join(self.save_images_dir, "%s" % self.save_ckpt_name), exist_ok=True)
        os.makedirs(os.path.join(self.save_models_dir, "%s" % self.save_ckpt_name), exist_ok=True)
        
        self.writer = tf.summary.create_file_writer(os.path.join(self.results_dir, self.tb_log_path))


        for epoch in tqdm(range(opt.epoch, opt.n_epochs),
                          # leave=False,
                          unit='epoch'):
            for i, batch in tqdm(enumerate(self.train_dataset),
                                 # leave=False,
                                 total=self.train_dataset_size,
                                 unit='batch'):

                # start_step_time = time()

                self.train_step(images=batch["Image"], epoch=epoch)

                # # Print log
                # self.print_log(i_epoch=epoch, i_step=i, step_duration=time() - start_step_time, condition='train')



            # Generate image
            self.generate_and_save_images(model=self.generator, epoch=epoch,test_input=self.seed)

            # Save model
            self.save_model(epoch=epoch)

    @tf.function
    def train_step(self, images, epoch):
        noise = tf.random.normal([opt.batch_size, opt.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))



        ###################
        ### TensorBoard ###
        ###################

        write_to_tensorboard_data = {"Loss_G": gen_loss,
                                     "Loss_D": disc_loss}

        self.write_to_tensorboard(data_dict=write_to_tensorboard_data, epoch=epoch, condition='train')


    def generate_and_save_images(self, model, epoch, test_input):
    
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        
        predictions = model(test_input, training=False)

        plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
            plt.imshow(predictions[i] * 0.5 + 0.5)
            plt.axis('off')

        final_save_images_path = '%s/%s/ep{%s}.png' % (self.save_images_dir, self.save_ckpt_name, epoch)
        plt.savefig(final_save_images_path)
        plt.close('all')

    def validate(self, epoch_done, is_save_image=True):
        pass
        


    def test(self, is_save_image=True):
        os.makedirs(os.path.join(self.save_images_dir, "%s" % self.save_ckpt_name), exist_ok=True)
        self.generator.build((1, 100))
        self.generator.summary()
        self.generator.load_weights(self.load_model_path)
        self.generate_and_save_images(self.generator, epoch=10000, test_input=tf.random.normal([1, 100]))
        

    def save_model(self, epoch):

        # Save model checkpoints
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            print("\nSave model to [%s] at %d epoch\n" % (self.save_ckpt_name, epoch))
            
            final_save_model_path = "%s/%s/generator_%s" % (self.save_models_dir, self.save_ckpt_name, epoch)
            self.generator.save_weights(final_save_model_path + ".h5")
            


        # Save latest model
        if epoch == (opt.n_epochs - 1):
            print("\nSave latest model to [%s]\n" % self.save_ckpt_name)

            final_save_model_path = "%s/%s/generator_%s" % (self.save_models_dir, self.save_ckpt_name, opt.n_epochs)
            self.generator.save_weights(final_save_model_path, save_format='h5')
            


    def write_to_tensorboard(self, data_dict, epoch, condition):
        if self.writer is not None:

            if condition.strip() in ["train", "Train", "TRAIN"]:
                with self.writer.as_default():
                    for (name, data) in data_dict.items():
                        tf.summary.scalar(name=name, data=data, step=epoch)


            elif condition.strip() in ["val", "Val", "VAL"]:
                pass
            else:
                print("Please specify condition: [\"train\", \"Train\", \"TRAIN\"] or [\"val\", \"Val\", \"VAL\"]")
        else:
            print("Writer is None!")



    def print_log(self, i_epoch, i_step, step_duration=None, epoch_duration=None, condition="train"):
        '''
        Print out logs specified in log_template
        Note:
            1. The keys in self.evaluation_metrics or self.running_evaluation_metrics_dict should be also in log_template
            2. Otherwise, the one missing in log_template won't be printed.
        :param i_epoch: <int> current epoch (start from 1)
        :param i_step: <int> current step (start from 0)
        :param step_duration: <float> duration for forwarding a batch (a step)
        :param epoch_duration: <float> duration for running an epoch
        :param condition: manually specify "train" or "validation or "test" for log

        '''

        show_result_at_step = self.show_train_result_at_step if self.show_train_result_at_step is not None else 1

        if i_step % show_result_at_step == 0:

            log_str = log_template["epoch"] % (datetime.now(), i_epoch) + "; "

            log_str += log_template["step"] % (i_step) + " {} => ".format(condition)

            # log_str += log_template["loss"] % ("total_loss", self.running_loss_dict["total_loss"]) + "; "

            # for k, v in self.running_evaluation_metrics_dict.items():

                # if k in log_template.keys():
                #     log_str += log_template[k] % (k, v) + "; "

            if step_duration is not None:
                log_str += " (" + log_template["step-duration"] % (
                    opt.batch_size / float(step_duration), step_duration) \
                           + ")"

            if epoch_duration is not None:
                log_str += " (" + log_template["epoch-duration"] % (epoch_duration) + ") "

                rest_of_time = ((opt.n_epochs - i_epoch) * epoch_duration)
                rest_of_time_in_min = rest_of_time / 60.0
                rot_hr = (rest_of_time_in_min // 60)
                rot_min = rest_of_time_in_min % 60

                log_str += "=> " + log_template["time-remaining"] % (rot_hr, rot_min)

            tqdm.write(log_str)


    def print_progress_bar(self, iteration, total, prefix='', suffix='', decimals=1, length=50, fill='=', forward_sample='>'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(int(100 * (iteration / float(total))))
        filled_length = int(length * iteration // total)
        forward_symbol = forward_sample if iteration < total else ''
        bar = fill * filled_length + forward_symbol + '-' * (length - filled_length)
        print('\r %s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()
            print()
