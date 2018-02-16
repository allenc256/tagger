import tensorflow as tf

from .feature import FEATURE_NAMES


def parse_example(proto, max_len=None):
    """
    Parses an example into a list.
    :param proto: the proto of the example to parse
    :param max_len: max length to pad/crop examples to
    :return: parsed tensors, one per feature, plus a dimension 0 tensor
        representing the length of the example
    """
    # parse proto
    parsed = tf.parse_single_example(
        proto, features={
            name: tf.VarLenFeature(tf.int64) for name in FEATURE_NAMES})

    # convert to dense tensors
    features = [
        tf.sparse_tensor_to_dense(parsed[name]) for name in FEATURE_NAMES]

    # determine example length
    ex_len = tf.shape(features[0])[0]

    if max_len:
        # pad to window_width (on the right)
        features = [
            tf.pad(f, [[0, tf.maximum(0, max_len - ex_len)]])
            for f in features]

        # crop to window_width (on the right)
        features = [f[:max_len] for f in features]
        ex_len = tf.minimum(ex_len, max_len)

    return features + [ex_len]


def dataset(filename, max_len=None, compression_type=None, limit=None,
            num_parallel_calls=1, shuffle_size=None, batch_size=None,
            repeat=False):
    """
    Builds a TFRecordDataset from a file.
    :param filename: the file containing the dataset
    :param max_len: max length to pad/crop examples to - must be specified if
        batching examples
    :param compression_type: compression type for the dataset
    :param limit: limit on number of examples to take from dataset
    :param num_parallel_calls: parallelization level for parsing
    :param shuffle_size: shuffle buffer size
    :param batch_size: batching size
    :param repeat: flag for if the dataset should repeat
    :return: built TFRecordDataset
    """
    if batch_size and not max_len:
        raise ValueError('max_len must be specified when batching')
    d = tf.data.TFRecordDataset(filename, compression_type=compression_type)
    if limit:
        d = d.take(limit)
    d = d.map(
        lambda p: parse_example(p, max_len=max_len),
        num_parallel_calls=num_parallel_calls)
    if shuffle_size:
        d = d.shuffle(shuffle_size)
    if repeat:
        d = d.repeat()
    if batch_size:
        d = d.batch(batch_size)
    return d
