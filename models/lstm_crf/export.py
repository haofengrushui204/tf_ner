"""Export model as a saved_model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
import json

import tensorflow as tf

from main import model_fn

root_dir = "/data/kongyy/nlp/tf_ner_guillaumegenthial/"
DATADIR = root_dir + 'example'

# DATADIR = '../../data/example'
PARAMS = root_dir + 'results/params.json'
MODELDIR = root_dir + 'results/model'


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    receiver_tensors = {'words': words, 'nwords': nwords}
    features = {'words': words, 'nwords': nwords}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    with Path(PARAMS).open() as f:
        params = json.load(f)

    params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
    params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
    params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
    params['glove'] = str(Path(DATADIR, 'w2v.npz'))

    estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)
    estimator.export_savedmodel(root_dir + 'saved_model', serving_input_receiver_fn)
