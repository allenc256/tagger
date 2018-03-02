import logging
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

import tagger.model.layers as layers
from tagger.model.dataset import FEATURE_NAMES

logger = logging


def _prepare_features(data_it):
    features = SimpleNamespace()
    features.ent_type, features.is_title, features.like_num, features.pos, \
        features.tag, features.target, features.text, input_len = \
        data_it.get_next()
    return features, input_len


def _prepare_embeddings(hparams, data_dict, word_embedding):
    embeddings = SimpleNamespace()

    # pre-trained text feature embedding
    logger.info('prepared word embedding: %s', word_embedding.shape)
    embeddings.text = tf.get_variable(
        'text_embedding', shape=word_embedding.shape, trainable=False,
        initializer=tf.constant_initializer(word_embedding),
        dtype=tf.float32)

    # trainable feature embeddings
    for feat_name in FEATURE_NAMES:
        if feat_name == 'text' or feat_name == 'target':
            continue
        emb_size = getattr(data_dict, feat_name).size()
        emb_dim = getattr(hparams.embedding_dims, feat_name)
        logger.info(
            'prepared trainable embedding %s: (%d, %d)', feat_name,
            emb_size, emb_dim)
        setattr(embeddings, feat_name, tf.get_variable(
            '%s_embedding' % feat_name, [emb_size, emb_dim]))

    return embeddings


class AttentionModel:
    def __init__(self, hparams, data_it, handle, data_dict, word_embedding):
        self.handle = handle
        self.data_dict = data_dict
        self.hparams = hparams

        # learning rate
        # N.B., decay is applied, so this needs to be saved w/ the model
        self.learning_rate = tf.get_variable(
            'learning_rate', shape=[],
            initializer=tf.constant_initializer(hparams.learning_rate),
            trainable=False)

        # training flag
        self.training = tf.placeholder(tf.bool, name='training')

        # prepare features/embeddings
        self.features, self.input_len = _prepare_features(data_it)
        embeddings = _prepare_embeddings(
            hparams, data_dict, word_embedding)

        # input mask
        input_mask = tf.sequence_mask(
            self.input_len, maxlen=tf.shape(self.features.text)[1],
            dtype=tf.float32)

        # embed features and concat
        embedded = []
        for name in FEATURE_NAMES:
            if name == 'target':
                continue
            f = getattr(self.features, name)
            e = getattr(embeddings, name)
            embedded.append(tf.nn.embedding_lookup(e, f))
        layer = tf.concat(embedded, axis=-1)

        # project to hidden_dim
        layer = tf.layers.dropout(
            layer, rate=hparams.dropout_rate, training=self.training)
        layer = tf.layers.dense(
            layer, hparams.hidden_dim, name='input_proj')

        # encoding layers
        for i in range(hparams.num_attention_layers):
            layer = layers.attention_encode(
                layer, input_mask, hparams.hidden_dim_ff, hparams.dropout_rate,
                self.training, name='encode_%d' % i)

        # dropout
        layer = tf.layers.dropout(
            layer, rate=hparams.dropout_rate, training=self.training)

        # logits: is a tag
        is_tag_logits = tf.layers.dense(layer, 1, name='tag')
        is_tag_logits = tf.squeeze(is_tag_logits, axis=-1, name='tag_logit')

        # logits: target class
        class_logits = tf.layers.dense(
            layer, data_dict.target.size(), name='class')

        # tag losses
        is_tag = tf.cast(tf.greater(self.features.target, 0), tf.float32)
        is_tag_losses = tf.nn.weighted_cross_entropy_with_logits(
            targets=is_tag, logits=is_tag_logits,
            pos_weight=hparams.tag_pos_weight)
        is_tag_losses *= input_mask
        is_tag_losses = tf.reduce_sum(is_tag_losses, axis=-1)

        # class losses
        class_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.features.target, logits=class_logits)
        class_losses *= is_tag
        class_losses = tf.reduce_sum(class_losses, axis=-1)

        # mean loss
        self.mean_loss_is_tag = tf.reduce_mean(is_tag_losses)
        self.mean_loss_class = tf.reduce_mean(class_losses)
        self.mean_loss = self.mean_loss_is_tag + self.mean_loss_class

        # estimates
        self.is_tag_est = tf.nn.sigmoid(is_tag_logits) > 0.5
        self.class_est = tf.argmax(tf.nn.softmax(class_logits), axis=-1)

        # optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gs = opt.compute_gradients(self.mean_loss)
        gs, vs = zip(*gs)
        gs, _ = tf.clip_by_global_norm(gs, hparams.grad_clip_norm)
        self.train_op = opt.apply_gradients(
            zip(gs, vs), global_step=self.global_step)


class RnnModel:
    def __init__(self, hparams, data_it, handle, data_dict, word_embedding):
        self.handle = handle
        self.data_dict = data_dict
        self.hparams = hparams

        # learning rate
        # N.B., decay is applied, so this needs to be saved w/ the model
        self.learning_rate = tf.get_variable(
            'learning_rate', shape=[],
            initializer=tf.constant_initializer(hparams.learning_rate),
            trainable=False)

        # training flag
        self.training = tf.placeholder(tf.bool, name='training')

        # prepare features/embeddings
        self.features, self.input_len = _prepare_features(data_it)
        embeddings = _prepare_embeddings(
            hparams, data_dict, word_embedding)

        # input mask
        input_mask = tf.sequence_mask(
            self.input_len, maxlen=tf.shape(self.features.text)[1],
            dtype=tf.float32)

        # embed features and concat
        embedded = []
        for name in FEATURE_NAMES:
            if name == 'target':
                continue
            f = getattr(self.features, name)
            e = getattr(embeddings, name)
            embedded.append(tf.nn.embedding_lookup(e, f))
        layer = tf.concat(embedded, axis=-1)

        # encode
        layer = layers.rnn(
            layer, self.input_len, hparams.hidden_dim,
            hparams.num_encoder_rnn_layers, hparams.dropout_rate, self.training,
            name='encode')

        # self-attention
        layer = layers.attention_gated(
            layer, layer, input_mask, hparams.dropout_rate, self.training)

        # memory
        layer = layers.rnn(
            layer, self.input_len, hparams.hidden_dim,
            hparams.num_memory_rnn_layers, hparams.dropout_rate, self.training,
            name='memory')

        # dropout
        layer = tf.layers.dropout(
            layer, rate=hparams.dropout_rate, training=self.training)

        # logits: is a tag
        is_tag_logits = tf.layers.dense(layer, 1, name='tag')
        is_tag_logits = tf.squeeze(is_tag_logits, axis=-1, name='tag_logit')

        # logits: target class
        class_logits = tf.layers.dense(
            layer, data_dict.target.size(), name='class')

        # tag losses
        is_tag = tf.cast(tf.greater(self.features.target, 0), tf.float32)
        is_tag_losses = tf.nn.weighted_cross_entropy_with_logits(
            targets=is_tag, logits=is_tag_logits,
            pos_weight=hparams.tag_pos_weight)
        is_tag_losses *= input_mask
        is_tag_losses = tf.reduce_sum(is_tag_losses, axis=-1)

        # class losses
        class_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.features.target, logits=class_logits)
        class_losses *= is_tag
        class_losses = tf.reduce_sum(class_losses, axis=-1)

        # mean loss
        self.mean_loss_is_tag = tf.reduce_mean(is_tag_losses)
        self.mean_loss_class = tf.reduce_mean(class_losses)
        self.mean_loss = self.mean_loss_is_tag + self.mean_loss_class

        # estimates
        self.is_tag_est = tf.nn.sigmoid(is_tag_logits) > 0.5
        self.class_est = tf.argmax(tf.nn.softmax(class_logits), axis=-1)

        # optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gs = opt.compute_gradients(self.mean_loss)
        gs, vs = zip(*gs)
        gs, _ = tf.clip_by_global_norm(gs, hparams.grad_clip_norm)
        self.train_op = opt.apply_gradients(
            zip(gs, vs), global_step=self.global_step)


class LearningRateDecayer:
    def __init__(self, model, max_patience):
        self.model = model
        self.patience = 0
        self.max_patience = max_patience
        self.best_loss = None

    def _reset(self, loss):
        self.best_loss = loss
        self.patience = 0

    def decay(self, loss):
        if self.best_loss is None or loss < self.best_loss:
            self._reset(loss)
        else:
            self.patience += 1
        if self.patience >= self.max_patience:
            new_lr = self.model.learning_rate.eval() / 2
            logger.info('decaying learning rate: %g', new_lr)
            tf.assign(self.model.learning_rate, new_lr).eval()
            self._reset(loss)


def evaluate(sess, model, data_handle, header=''):
    # counters
    is_tag_cor_count = 0
    is_tag_est_count = 0
    is_tag_act_count = 0
    is_tag_tot_loss = 0
    class_cor_count = 0
    class_tot_count = 0
    class_tot_loss = 0
    ex_count = 0

    for _ in range(model.hparams.eval_steps):
        text, input_len, target, is_tag_est, class_est, \
            is_tag_loss, class_loss = sess.run(
                [model.features.text, model.input_len, model.features.target,
                 model.is_tag_est, model.class_est, model.mean_loss_is_tag,
                 model.mean_loss_class],
                feed_dict={model.handle: data_handle, model.training: False})

        is_tag_tot_loss += is_tag_loss
        class_tot_loss += class_loss

        for i in range(text.shape[0]):
            # write example dump header to log
            ex_count += 1
            if ex_count <= model.hparams.dump_examples:
                logger.debug('%s example %d:', header, ex_count)

            for j in range(input_len[i]):
                act = target[i, j]
                est = class_est[i, j] if is_tag_est[i, j] else 0

                # update counters for localizing tags
                if act > 0 and est > 0:
                    is_tag_cor_count += 1
                if act > 0:
                    is_tag_act_count += 1
                if est > 0:
                    is_tag_est_count += 1

                # update counters for tag accuracy
                if act > 0 and est > 0:
                    class_tot_count += 1
                    if act == est:
                        class_cor_count += 1

                # write example dump to log
                if ex_count <= model.hparams.dump_examples:
                    logger.debug(
                        '%20.20s %20.20s %20.20s',
                        model.data_dict.text.value(text[i, j]),
                        model.data_dict.target.value(act or 0) or '-',
                        model.data_dict.target.value(est or 0) or '-')

    # compute stats
    pre = is_tag_cor_count / np.maximum(is_tag_est_count, 1)
    rec = is_tag_cor_count / np.maximum(is_tag_act_count, 1)
    f1 = (2 * pre * rec) / (pre + rec + 1e-10)
    acc = class_cor_count / np.maximum(class_tot_count, 1)
    mean_is_tag_loss = is_tag_tot_loss / model.hparams.eval_steps
    mean_class_loss = class_tot_loss / model.hparams.eval_steps

    return f1, acc, mean_is_tag_loss, mean_class_loss


def dump_statistics():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        if shape.is_fully_defined():
            for dim in shape:
                variable_parameters *= dim.value
            logger.info(
                'parameters for "%s": %d' % (variable.name, variable_parameters))
            total_parameters += variable_parameters
        else:
            logger.info('parameters for "%s": ?' % variable.name)
    logger.info('total parameters: %d' % total_parameters)
