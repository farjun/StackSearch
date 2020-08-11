import os
from datetime import datetime

from tqdm import tqdm
from dataprocess.api import resolve_data_set
import tensorflow as tf

from models.api import getNNHashEncoder

class TfWriter(object):
    def __init__(self):
        now_as_string = datetime.now().strftime("%m_%d_%H_%M_%S")  # current date and time
        writer_path = os.path.join("summary", "train", now_as_string)
        self.writer = tf.summary.create_file_writer(writer_path)
        print(f"writer_path {writer_path}")

    def reprortProgress(self, loss, step):
        with self.writer.as_default():
            tf.summary.scalar("loss", loss.result(), step=step)

def getTrainStep(model, optimizer = tf.keras.optimizers.Adam()):

    @tf.function
    def train_step(data, metric):
        loss_object = tf.keras.losses.MeanSquaredError()
        reg_loss_object = tf.keras.losses.MeanSquaredError()
        reg_lambda = 1e-2
        with tf.GradientTape(persistent=True) as tape:
            encoded_data = model.encode(data, training=True)
            same = model.decode(encoded_data, training=True)
            ae_loss = loss_object(data, same)
            reg_loss = reg_lambda * (-1 * reg_loss_object(tf.zeros_like(encoded_data), encoded_data))
            loss = ae_loss + reg_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        metric(ae_loss)

    return train_step

def train_yabadaba(epochs=1, epochs_offset=0, progress_per_step=1,
                   save_result_per_epoch=5, restore_last=True, dataset_type: str = None):
    ds = resolve_data_set(dataset_type)
    nnHashEncoder = getNNHashEncoder(restore_last)
    train_step = getTrainStep(nnHashEncoder.model, nnHashEncoder.optimizer)
    writer = TfWriter()
    step = 0
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    for epoch in tqdm(range(epochs_offset, epochs + epochs_offset), desc="train epochs"):
        if epoch % save_result_per_epoch == 0:
            nnHashEncoder.save()

        for data in ds:
            train_step(data, train_loss)
            if step % progress_per_step == 0:
                writer.reprortProgress(train_loss, step)

            step += 1

    nnHashEncoder.save()
