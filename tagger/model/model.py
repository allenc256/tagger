import logging
from types import SimpleNamespace

import tensorflow as tf

import tagger.model.layers
from tagger.model.dataset import FEATURE_NAMES


logger = logging


class Model:
    def __init__(self, hparams, data_it, handle, data_dict, word_embedding):
        self.handle = handle
        self.data_dict = data_dict
        self.hparams = hparams

        # training flag
        self.training = tf.placeholder(tf.bool, name='training')

        # prepare features/embeddings
        self.features, self.input_len = self._prepare_features(data_it)
        embeddings = self._prepare_embeddings(
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
        layer = tagger.model.layers.rnn(
            layer, self.input_len, hparams.hidden_dim,
            hparams.num_encoder_layers, hparams.dropout_rate, self.training,
            name='encode')
        # layer = tf.layers.dropout(
        #     layer, rate=hparams.dropout_rate, training=self.training)
        # layer = tf.layers.dense(layer, hparams.hidden_dim, name='encode_post')

        # memory
        layer = tagger.model.layers.rnn(
            layer, self.input_len, hparams.hidden_dim,
            hparams.num_memory_layers, hparams.dropout_rate, self.training,
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

        # class losses
        class_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.features.target, logits=class_logits)
        class_losses *= is_tag

        # mean loss
        self.mean_loss_is_tag = tf.reduce_mean(is_tag_losses)
        self.mean_loss_class = tf.reduce_mean(class_losses)
        self.mean_loss = self.mean_loss_is_tag + self.mean_loss_class

        # estimates
        self.is_tag_est = tf.nn.sigmoid(is_tag_logits) > 0.5
        self.class_est = tf.argmax(tf.nn.softmax(class_logits), axis=-1)

        # optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        opt = tf.train.AdamOptimizer(hparams.learning_rate)
        gs = opt.compute_gradients(self.mean_loss)
        gs, vs = zip(*gs)
        gs, _ = tf.clip_by_global_norm(gs, hparams.grad_clip_norm)
        self.train_op = opt.apply_gradients(
            zip(gs, vs), global_step=self.global_step)

    def eval(self, sess, data_handle):
        text, input_len, target, is_tag_est, class_est = sess.run(
            [self.features.text, self.input_len, self.features.target,
             self.is_tag_est, self.class_est],
            feed_dict={self.handle: data_handle, self.training: False})

        for i in range(text.shape[0]):
            logger.debug('example %d:', i)
            for j in range(input_len[i]):
                target_est = '-'
                if is_tag_est[i, j]:
                    target_est = self.data_dict.target.value(class_est[i, j])
                logger.debug(
                    '%20.20s %20.20s: %20.20s',
                    self.data_dict.text.value(text[i, j]),
                    self.data_dict.target.value(target[i, j]) or '-',
                    target_est)

    @staticmethod
    def _prepare_features(data_it):
        features = SimpleNamespace()
        features.ent_type, features.is_title, features.like_num, features.pos, \
            features.tag, features.target, features.text, input_len = \
            data_it.get_next()
        return features, input_len

    @staticmethod
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
