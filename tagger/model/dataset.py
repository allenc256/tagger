import tensorflow as tf


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
            num_parallel_calls=1, shuffle_size=None, prefetch_size=None,
            batch_size=None, repeat=False):
    """
    Builds a TFRecordDataset from a file.
    :param filename: the file containing the dataset
    :param max_len: max length to pad/crop examples to - must be specified if
        batching examples
    :param compression_type: compression type for the dataset
    :param limit: limit on number of examples to take from dataset
    :param num_parallel_calls: parallelization level for parsing
    :param shuffle_size: shuffle buffer size
    :param prefetch_size: prefetch buffer size
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
    if prefetch_size:
        d = d.prefetch(prefetch_size)
    if shuffle_size:
        d = d.shuffle(shuffle_size)
    if repeat:
        d = d.repeat()
    if batch_size:
        d = d.batch(batch_size)
    return d


class FeatureDictionary:
    """
    Bi-directional dictionary which maps values to/from integers for a
    particular feature.
    """
    def __init__(self):
        # ID zero always reserved
        self._id_to_value = [None]
        self._value_to_id = {None: 0}

    def id(self, value):
        """
        Get the ID corresponding to a feature value, or create a new ID if the
        value hasn't been seen before.
        :param value: the feature value
        :return: the corresponding ID
        """
        fid = self._value_to_id.get(value)
        if fid is None:
            fid = len(self._id_to_value)
            self._value_to_id[value] = fid
            self._id_to_value.append(value)
        return fid

    def value(self, fid):
        """
        Get the value corresponding to a feature ID.
        :param fid: the feature ID
        :return: the corresponding value
        """
        return self._id_to_value[fid]

    def size(self):
        """
        Return the number of values in the dictionary.
        :return: number of values
        """
        return len(self._id_to_value)

    def to_json(self):
        """
        Create JSON representation.
        :return: JSON representation (callable with json.dump())
        """
        return self._id_to_value

    def from_json(self, json):
        """
        Load from JSON representation.
        :param json: JSON reprentation
        """
        self._id_to_value = json
        self._value_to_id = {v: i for i, v in enumerate(self._id_to_value)}


class DatasetDictionary:
    """
    A collection of feature dictionaries for a single dataset.
    """
    def __init__(self):
        # context words
        self.text = FeatureDictionary()
        # link target (0 for None)
        self.target = FeatureDictionary()
        # spacy NER entity type
        self.ent_type = FeatureDictionary()
        # spacy is_title flag
        self.is_title = FeatureDictionary()
        # spacy like_num flag
        self.like_num = FeatureDictionary()
        # spacy POS tag
        self.pos = FeatureDictionary()
        # spacy POS fine-grained tag
        self.tag = FeatureDictionary()

    def to_json(self):
        """
        Create JSON reprentation.
        :return: JSON representation (callable with json.dump())
        """
        return {fn: fd.to_json() for fn, fd in self.__dict__.items()}

    def from_json(self, json):
        """
        Load JSON representation.
        :param json: JSON reprentation
        """
        for fn, fd in json.items():
            getattr(self, fn).from_json(fd)


# N.B., some code relies on deterministic order here, so this needs to be sorted
FEATURE_NAMES = sorted(DatasetDictionary().__dict__.keys())