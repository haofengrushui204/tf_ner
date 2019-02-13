"""Export model as a saved_model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
import json

import tensorflow as tf

from main_ner_2phase import model_fn

root_dir = "/data/kongyy/nlp/tf_ner_guillaumegenthial/"


# DATADIR = root_dir + 'example'

# DATADIR = '../../data/example'


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    ptags = tf.placeholder(dtype=tf.float32, shape=[None,None], name='ptags')
    receiver_tensors = {'words': words, 'nwords': nwords, "ptags": ptags}
    features = {'words': words, 'nwords': nwords, "ptags": ptags}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("usage: python export-ner-2phase.py opinion_id")
        sys.exit(0)
    opinion_id = sys.argv[1]
    DATADIR = root_dir + 'example/{}/'.format(opinion_id)

    PARAMS = root_dir + 'results_{}/params.json'.format(opinion_id)
    MODELDIR = root_dir + 'results_{}/model'.format(opinion_id)

    with Path(PARAMS).open() as f:
        params = json.load(f)

    params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
    params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
    params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
    params['glove'] = str(Path(DATADIR, 'w2v_{}.npz'.format(opinion_id)))

    estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)
    estimator.export_savedmodel(root_dir + 'saved_model_{}'.format(opinion_id), serving_input_receiver_fn)
