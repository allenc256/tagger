import tensorflow as tf
from types import SimpleNamespace

from .dataset import FEATURE_NAMES


class Model:
    def __init__(self, hparams, data_it, data_dict, word_embedding):
        features = self._prepare_features(data_it)

        embeddings = self._prepare_embeddings(
            hparams, data_dict, word_embedding)

        # embed features and concat
        embedded = []
        for name in FEATURE_NAMES:
            f = getattr(features, name)
            e = getattr(embeddings, name)
            embedded.append(tf.nn.embedding_lookup(e, f))
        inputs = tf.concat(embedded, axis=-1)

    @staticmethod
    def _prepare_features(data_it):
        features = SimpleNamespace()

        features.ent_type, features.is_title, features.like_num, features.pos, \
            features.tag, features.target, features.text, ex_len = \
            data_it.get_next()

        return features

    @staticmethod
    def _prepare_embeddings(hparams, data_dict, word_embedding):
        embeddings = SimpleNamespace()

        # pre-trained text feature embedding
        embeddings.text = tf.get_variable(
            'text_embedding',
            initializer=tf.constant_initializer(word_embedding),
            trainable=False)

        # trainable feature embeddings
        for feat_name in FEATURE_NAMES:
            if feat_name == 'text':
                continue
            emb_size = getattr(data_dict, feat_name).size()
            emb_dim = getattr(hparams.embedding_dims, feat_name)
            setattr(embeddings, feat_name, tf.get_variable(
                '%s_embedding' % feat_name, [emb_size, emb_dim]))

        return embeddings





    def train(self):
        pass
