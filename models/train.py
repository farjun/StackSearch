import os
from datetime import datetime
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from dataprocess.api import resolve_data_set
import tensorflow as tf
import tensorflow_probability as tfp
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

    def reprortProgressMany(self, stuff: list, step):
        with self.writer.as_default():
            for toReport in stuff:
                tf.summary.scalar(toReport.name, toReport.result(), step=step)

    def reprortProgressManyWithNameScope(self, stuff: dict, step: int):
        with self.writer.as_default():
            for nameScope, toReportMany in stuff.items():
                with tf.name_scope(nameScope):
                    for toReport in toReportMany:
                        tf.summary.scalar(toReport.name, toReport.result(), step=step)


def getDiscriminatorLoss():
    discriminator_research_loss_metric = tf.keras.metrics.Mean(name='discriminator_research_loss')

    def discriminator_research_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        discriminator_research_loss_metric(total_loss)
        return total_loss

    return discriminator_research_loss, [discriminator_research_loss_metric]


def getGeneratorLoss(lossObject):
    generatorVsDiscriminatorLosssReport = tf.keras.metrics.Mean(name='gen-vs-discriminator-train_loss')
    generatorReconstructionLosssReport = tf.keras.metrics.Mean(name='gen-reconstruction-train_loss')

    def generator_research_loss(fake_output, data, genOutput):
        reconstructionLoss = lossObject(data, genOutput)
        crossEntropyLoss = cross_entropy(tf.ones_like(fake_output), fake_output)
        generatorReconstructionLosssReport(reconstructionLoss)
        generatorVsDiscriminatorLosssReport(crossEntropyLoss)
        lambda1 = HParams.RECONSTRUCTION_LOSS_LAMBDA
        lambda2 = HParams.CROSS_ENTROPY_LOSS_LAMBDA
        return lambda1 * reconstructionLoss + lambda2 * crossEntropyLoss

    return generator_research_loss, [generatorReconstructionLosssReport, generatorVsDiscriminatorLosssReport]


def getTrainStep(model, discriminator, noiseFunction=None):
    # optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # derivetive by
    reconstructionLoss = tf.keras.losses.MeanSquaredError(name='autoencoder_reconstruction_loss')
    genTrainLoss, toReportGen = getGeneratorLoss(reconstructionLoss)
    discTrainLoss, toReportDisc = getDiscriminatorLoss()

    # prob distributions
    randomVecDistribution = tfp.distributions.Bernoulli(
        probs=tf.constant(0.5, shape=(HParams.BATCH_SIZE, HParams.OUTPUT_DIM)))

    @tf.function
    def train_step(data: tf.Tensor):
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape() as disc_tape:
            encoded_data = model.encode(data, training=True)
            genOutput = model.decode(encoded_data, training=True)
            randomVec = randomVecDistribution.sample()
            fake_vec_output = discriminator(encoded_data, training=True)
            real_vec_output = discriminator(randomVec, training=True)

            generator_loss = genTrainLoss(fake_vec_output, data, genOutput)
            discriminator_loss = discTrainLoss(real_vec_output, fake_vec_output)

        autoencoder_gradients = gen_tape.gradient(generator_loss, model.trainable_variables)
        discriminator_gradients = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(autoencoder_gradients, model.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return train_step, {"Gen": toReportGen, "Disc": toReportDisc}


def getTrainStepNotGan(model):
    # optimizers
    optimizer = tf.keras.optimizers.Adam(HParams.LR)

    # derivetive by
    reconstructionLossObject = tf.keras.losses.MeanSquaredError(name='autoencoder_reconstruction_loss')
    binaryLossObject = tf.keras.losses.MeanSquaredError(name='autoencoder_reconstruction_loss')

    reconstructionLosssReport = tf.keras.metrics.Mean(name='reconstruction-train_loss')
    binaryLossReport = tf.keras.metrics.Mean(name='binary-train_loss')
    lossReport = tf.keras.metrics.Mean(name='total-train_loss')

    @tf.function
    def train_step(data_noised, data_target):
        with tf.GradientTape(persistent=True) as gen_tape:
            encoded_data = model.encode(data_noised, training=True)
            genOutput = model.decode(encoded_data, training=True)
            reconstructionLoss = reconstructionLossObject(genOutput, data_target)
            binaryLoss = binaryLossObject(encoded_data, tf.constant(0.5, shape=encoded_data.shape))
            reconstructionLosssReport(reconstructionLoss)
            binaryLossReport(binaryLoss)
            loss = HParams.RECONSTRUCTION_LOSS_LAMBDA * reconstructionLoss
            lossReport(loss)

        autoencoder_gradients = gen_tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(autoencoder_gradients, model.trainable_variables))

    return train_step, {"Gen": [reconstructionLosssReport, binaryLossReport, lossReport]}


def train_yabadaba(epochs=1, epochs_offset=0, progress_per_step=1,
                   save_result_per_epoch=5, restore_last=False, dataset_type: str = HParams.DATASET):
    ds = resolve_data_set(dataset_type, amount_to_drop=HParams.AMOUNT_TO_DROP, amount_to_swap=HParams.AMOUNT_TO_SWAP)
    nnHashEncoder = getNNHashEncoder(restore_last, skip_discriminator=True)
    # train_step, reportStuff = getTrainStep(nnHashEncoder.model, nnHashEncoder.discriminator)
    train_step, reportStuff = getTrainStepNotGan(nnHashEncoder.model)
    writer = TfWriter()

    step = 0
    for epoch in tqdm(range(epochs_offset, epochs + epochs_offset), desc="train epochs"):
        for data_noised, data_target in ds:
            train_step(data_noised, data_target)
            if step % progress_per_step == 0:
                writer.reprortProgressManyWithNameScope(reportStuff, step)

            step += 1

        nnHashEncoder.save()
        for toReportMany in reportStuff.values():
            for toReport in toReportMany:
                toReport.reset_states()


def train_embedding_word2vec(numOfWordsToDrop=0):
    from dataprocess.parser import XmlParser
    from gensim.models import Word2Vec
    xmlParser = XmlParser(HParams.filePath)
    tmp = ord('z') - ord('a') + 7
    model = Word2Vec(size=(HParams.MAX_SENTENCE_DIM * tmp), window=10, min_count=1, workers=4,
                     negative=numOfWordsToDrop)
    model.build_vocab((xmlParser.getSentsGenerator())())
    model.train((xmlParser.getSentsGenerator())(), total_examples=model.corpus_count, epochs=model.iter)
    model.save(os.path.join(HParams.embeddingFilePath, f"word2v_embedding_{numOfWordsToDrop}"))


def train_embedding_word2vec_new(*args, **kwargs):
    from dataprocess.parser import XmlParser
    from gensim.models import Word2Vec
    xmlParser = XmlParser(HParams.filePath)
    vec_size = 200
    model = Word2Vec(*args, size=(vec_size), window=10, min_count=1, workers=4, **kwargs)
    model.build_vocab((xmlParser.getSentsGenerator())())
    model.train((xmlParser.getSentsGenerator())(), total_examples=model.corpus_count, epochs=model.iter)
    model.save(os.path.join(HParams.embeddingFilePath, f"word2v_embedding"))


def train_embedding_doc2vec(numOfWordsToDrop=0):
    from dataprocess.parser import XmlParser
    from gensim.models import Doc2Vec
    xmlParser = XmlParser(HParams.filePath)  # TODO SET TO FULL DATA
    tmp = ord('z') - ord('a') + 7
    model = Doc2Vec(vector_size=(HParams.MAX_SENTENCE_DIM * tmp), min_count=2, epochs=40, negative=numOfWordsToDrop)
    model.build_vocab((xmlParser.getSentsGenerator(tagged=True))())
    model.train((xmlParser.getSentsGenerator(tagged=True))(), total_examples=model.corpus_count, epochs=model.iter)
    model.save(os.path.join(HParams.embeddingFilePath, f"doc2v_embedding_{numOfWordsToDrop}"))
