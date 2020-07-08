# from tqdm.auto import tqdm
from tqdm import tqdm
import os
from datetime import datetime
import tensorflow as tf
from Features.FeatureExtractors import FeatureExtractor, HistogramFeatureExtractor
import random
import models.BasicFCN

#### Setup ####
output_dim = 10
featureExtractor = HistogramFeatureExtractor()
model = models.BasicFCN.get_FCN_model(featureExtractor.get_feature_dim(), output_dim)
optimizer = tf.keras.optimizers.Adam()


### Done Setup ###

def example_get_data_generator(featureExtractor: FeatureExtractor):
    size = 1000
    stop_prob = 0.1
    words = []
    random_chr = lambda: chr(random.randint(ord('a'), ord('z')))
    for i in range(size):
        word = random_chr()
        while random.random() > stop_prob:
            word += random_chr()
        words.append(word)
    as_features = featureExtractor.get_feature_batch(words)

    def to_generator():
        for features in as_features:
            yield features

    return to_generator


def get_data_set_example():
    ds = tf.data.Dataset.from_generator(
        example_get_data_generator(HistogramFeatureExtractor()),
        (tf.int32))
    ds = ds \
        .cache() \
        .batch(4) \
        .shuffle(10000) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_ckpt_manager(restore_last=True):
    checkpoint_path = f"./checkpoints/train_{featureExtractor.get_feature_dim()}_{output_dim}"
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if restore_last:
        save_path = ckpt_manager.latest_checkpoint or tf.train.latest_checkpoint(checkpoint_path)
        if save_path is not None:
            ckpt.restore(save_path)
            print('Latest checkpoint restored!!')
    return ckpt_manager


def reprortProgress(writer, metric, step):
    with writer.as_default():
        tf.summary.scalar("loss", metric.result(), step=step)
    # print(".",end="")


@tf.function
def train_step(data, metric):
    loss_object = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape(persistent=True) as tape:
        same = model(data, training=True)
        loss = loss_object(data, same)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    metric(loss)


def train(epochs=1, epochs_offset=0, progress_per_step=1,
          save_result_per_epoch=5, restore_last=True):
    ds = get_data_set_example()
    ckpt_manager = get_ckpt_manager(restore_last)
    now_as_string = datetime.now().strftime("%m_%d_%H_%M_%S")  # current date and time
    writer_path = os.path.join("summary", "train", now_as_string)
    writer = tf.summary.create_file_writer(writer_path)
    print(f"writer_path {writer_path}")

    step = 0
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    for epoch in tqdm(range(epochs_offset, epochs + epochs_offset), desc="train epochs"):
        if epoch % save_result_per_epoch == 0:
            ckpt_save_path = ckpt_manager.save()
        for data in ds:
            train_step(data, train_loss)
            if step % progress_per_step == 0:
                reprortProgress(writer, train_loss, step)
            step += 1
    ckpt_save_path = ckpt_manager.save()


if __name__ == '__main__':
    train(epochs=1000, restore_last=False, progress_per_step=100)
