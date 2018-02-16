class FeatureDictionary:
    """
    Bi-directional dictionary which maps values to/from integers for a
    particular feature.
    """
    def __init__(self):
        # ID zero always reserved
        self._id_to_value = [None]
        self._value_to_id = {None: 0}

    def get_id(self, value):
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

    def get_value(self, fid):
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


FEATURE_NAMES = sorted(DatasetDictionary().__dict__.keys())
