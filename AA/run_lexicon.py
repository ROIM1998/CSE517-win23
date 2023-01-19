import numpy as np

from utils.data_utils import get_raw_data, text_to_feature, load_lexicon
from models.model import LexiconModel

if __name__ == '__main__':
    negatives, positives = load_lexicon('lexicon')
    model = LexiconModel(negatives, positives)
    data = get_raw_data('data/txt_sentoken')
    data = [(text_to_feature(d[0]), d[1]) for d in data]
    prediction = [model.predict(text) for text, label in data]
    labels = np.array([label for text, label in data])
    is_positive = np.array([int(v > 0) for v in prediction])
    accuracy = np.mean(is_positive == labels)