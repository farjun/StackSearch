import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from dataprocess.api import resolve_data_set
import tensorflow as tf

from hparams import HParams
from models.api import getNNHashEncoder

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class TfWriter(object):
    def __init__(self):
        now_as_string = datetime.now().strftime("%m_%d_%H_%M_%S")  # current date and time
        writer_path = os.path.join("summary", "train", now_as_string)
        self.writer = tf.summary.create_file_writer(writer_path)
        print(f"writer_path {writer_path}")

    def reprortProgress(self, loss, step):
        with self.writer.as_default():
            tf.summary.scalar("loss", loss.result(), step=step)

    def reprortProgressMany(self, stuff, step):
        with self.writer.as_default():
            for toReport in stuff:
                tf.summary.scalar(toReport.name, toReport.result(), step=step)


def discriminator_research_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_research_loss(fake_output, data, same):
    #todo add the reconstruction loss to the encoder
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def getTrainStep(model, discriminator):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    generator_rec_train_loss = tf.keras.metrics.Mean(name='autoencoder_reconstruction_loss')

    #report stuff
    generator_train_loss = tf.keras.metrics.Mean(name='gen-train_loss')
    discriminator_train_loss = tf.keras.metrics.Mean(name='disc-train_loss')
    generator_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='gen-train_accuracy')
    discriminator_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='disc-train_accuracy')

    @tf.function
    def train_step(data, metric):
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape() as disc_tape:
            encoded_data = model.encode(data, training=True)
            same = model.decode(encoded_data, training=True)

            randomVec = np.random.choice([0, 1], size=(HParams.BATCH_SIZE,HParams.OUTPUT_DIM))

            fake_vec_output = discriminator(encoded_data, training=True)
            real_vec_output = discriminator(randomVec, training=True)

            generator_loss = generator_research_loss(fake_vec_output, data, same)
            discriminator_loss = discriminator_research_loss(real_vec_output, fake_vec_output)


        autoencoder_gradients = gen_tape.gradient(generator_loss, model.trainable_variables)
        discriminator_gradients = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(autoencoder_gradients, model.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        generator_train_loss(generator_loss)
        discriminator_train_loss(discriminator_loss)

        generator_train_accuracy.update_state(tf.zeros_like(real_vec_output), real_vec_output)
        discriminator_train_accuracy.update_state(tf.ones_like(real_vec_output), real_vec_output)

    return train_step, [generator_train_loss, discriminator_train_loss, generator_train_accuracy, discriminator_train_accuracy]


def train_yabadaba(epochs=1, epochs_offset=0, progress_per_step=1,
                   save_result_per_epoch=5, restore_last=True, dataset_type: str = 'partial_titles'):
    ds = resolve_data_set(dataset_type)
    nnHashEncoder = getNNHashEncoder(restore_last)
    train_step, reportStuff = getTrainStep(nnHashEncoder.model, nnHashEncoder.discriminator)
    writer = TfWriter()
    step = 0
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    for epoch in tqdm(range(epochs_offset, epochs + epochs_offset), desc="train epochs"):
        if epoch % save_result_per_epoch == 0:
            nnHashEncoder.save()

        for data in ds:
            train_step(data, train_loss)
            if step % progress_per_step == 0:
                writer.reprortProgressMany(reportStuff, step)

            step += 1

        for toReport in reportStuff:
            toReport.reset_states()

    nnHashEncoder.save()


def train_embedding_word2vec():
    from dataprocess.parser import XmlParser
    from gensim.models import Word2Vec
    xmlParser = XmlParser(HParams.filePath)  #TODO SET TO FULL DATA
    tmp = ord('z') - ord('a') + 7
    model = Word2Vec(size=(HParams.MAX_SENTENCE_DIM * tmp), window=10, min_count=1, workers=4)
    model.build_vocab((xmlParser.getSentsGenerator())())
    model.train((xmlParser.getSentsGenerator())(), total_examples=model.corpus_count, epochs=model.iter)
    model.save(HParams.word2vecFilePath)


def train_embedding_doc2vec():
    from dataprocess.parser import XmlParser
    from gensim.models import Doc2Vec
    xmlParser = XmlParser(HParams.filePath)  #TODO SET TO FULL DATA
    tmp = ord('z') - ord('a') + 7
    model = Doc2Vec(vector_size=(HParams.MAX_SENTENCE_DIM * tmp), min_count=2, epochs=40)
    model.build_vocab((xmlParser.getSentsGenerator(tagged=True))())
    model.train((xmlParser.getSentsGenerator(tagged=True))(), total_examples=model.corpus_count, epochs=model.iter)
    model.save(HParams.doc2vecFilePath)
