# -*- coding:utf-8 -*-
"""
@file name: main.py
Created on 2019/5/14
@author: kyy_b
@desc:
"""
import functools
import json
import logging
from pathlib import Path
import sys

from models.chars_conv_dilated_conv import tf_utils
import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1

from models.chars_conv_dilated_conv.masked_conv import masked_conv1d_and_max

DATADIR = '../../data/example'

# Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),  # (words, nwords)
               ([None, None], [None])),  # (chars, nchars)
              [None])  # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('<pad>', 0),
                 ('<pad>', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def feature_layers(embeddings, reuse=True):
    """
    基于空洞卷积的特征提取
    :param embeddings:
    :return:
    """
    block_unflat_scores = []

    with tf.variable_scope("feature_layers", reuse=reuse):
        # input_list = [embeddings]
        input_size = params["dim_chars"] + params["dim"]  # word-embedding + char-embedding
        # if self.use_characters:
        #     char_embeddings_masked = tf.multiply(self.char_embeddings, tf.expand_dims(self.input_mask, 2))
        #     input_list.append(char_embeddings_masked)
        #     input_size += self.char_size

        initial_filter_width = params["layers_map"][0][1]['width']
        initial_num_filters = params["layers_map"][0][1]['filters']
        filter_shape = [1, initial_filter_width, input_size, initial_num_filters]
        initial_layer_name = "conv0"

        if not reuse:
            print("Adding initial layer %s: width: %d; filters: %d" % (
                initial_layer_name, initial_filter_width, initial_num_filters))

        input_feats = embeddings
        input_feats_expanded = tf.expand_dims(input_feats, 1)
        # input_feats_expanded_drop = tf.nn.dropout(input_feats_expanded, params["input_dropout_keep_prob"])
        print("input feats expanded drop", input_feats_expanded.get_shape())

        # first projection of embeddings
        w = tf_utils.initialize_weights(filter_shape, initial_layer_name + "_w", init_type='xavier', gain='relu')
        b = tf.get_variable(initial_layer_name + "_b", initializer=tf.constant(0.01, shape=[initial_num_filters]))
        conv0 = tf.nn.conv2d(input_feats_expanded, w, strides=[1, 1, 1, 1], padding="SAME", name=initial_layer_name)
        h0 = tf_utils.apply_nonlinearity(tf.nn.bias_add(conv0, b), 'relu')

        initial_inputs = [h0]
        last_dims = initial_num_filters

        # Stacked atrous convolutions
        last_output = tf.concat(axis=3, values=initial_inputs)

        for block in range(params["repeats"]):
            print("last out shape", last_output.get_shape())
            print("last dims", last_dims)
            hidden_outputs = []
            total_output_width = 0
            reuse_block = (block != 0 and params["share_repeats"]) or reuse
            block_name_suff = "" if params["share_repeats"] else str(block)
            inner_last_dims = last_dims
            inner_last_output = last_output
            with tf.variable_scope("block" + block_name_suff, reuse=reuse_block):
                for layer_name, layer in params["layers_map"]:
                    dilation = layer['dilation']
                    filter_width = layer['width']
                    num_filters = layer['filters']
                    initialization = layer['initialization']
                    take_layer = layer['take']
                    if not reuse:
                        print("Adding layer %s: dilation: %d; width: %d; filters: %d; take: %r" % (
                            layer_name, dilation, filter_width, num_filters, take_layer))
                    with tf.name_scope("atrous-conv-%s" % layer_name):
                        # [filter_height, filter_width, in_channels, out_channels]
                        filter_shape = [1, filter_width, inner_last_dims, num_filters]
                        w = tf_utils.initialize_weights(filter_shape, layer_name + "_w", init_type=initialization,
                                                        gain=params["nonlinearity"], divisor=params["num_classes"])
                        b = tf.get_variable(layer_name + "_b", initializer=tf.constant(
                            0.0 if initialization == "identity" or initialization == "varscale" else 0.001,
                            shape=[num_filters]))
                        # h = tf_utils.residual_layer(inner_last_output, w, b, dilation, self.nonlinearity, self.batch_norm, layer_name + "_r",
                        #                             self.batch_size, max_seq_len, self.res_activation, self.training) \
                        #     if last_output != input_feats_expanded_drop \
                        #     else tf_utils.residual_layer(inner_last_output, w, b, dilation, self.nonlinearity, False, layer_name + "_r",
                        #                             self.batch_size, max_seq_len, 0, self.training)

                        conv = tf.nn.atrous_conv2d(inner_last_output, w, rate=dilation, padding="SAME",
                                                   name=layer_name)
                        conv_b = tf.nn.bias_add(conv, b)
                        h = tf_utils.apply_nonlinearity(conv_b, params["nonlinearity"])

                        # so, only apply "take" to last block (may want to change this later)
                        if take_layer:
                            hidden_outputs.append(h)
                            total_output_width += num_filters
                        inner_last_dims = num_filters
                        inner_last_output = h

                h_concat = tf.concat(axis=3, values=hidden_outputs)
                last_output = tf.nn.dropout(h_concat, params["middle_dropout_keep_prob"])
                last_dims = total_output_width

                h_concat_squeeze = tf.squeeze(h_concat, [1])
                h_concat_flat = tf.reshape(h_concat_squeeze, [-1, total_output_width])

                # Add dropout
                with tf.name_scope("hidden_dropout"):
                    h_drop = tf.nn.dropout(h_concat_flat, params["hidden_dropout_keep_prob"])

                def do_projection():
                    # Project raw outputs down
                    with tf.name_scope("projection"):
                        projection_width = int(total_output_width / (2 * len(hidden_outputs)))
                        w_p = tf_utils.initialize_weights([total_output_width, projection_width], "w_p",
                                                          init_type="xavier")
                        b_p = tf.get_variable("b_p", initializer=tf.constant(0.01, shape=[projection_width]))
                        projected = tf.nn.xw_plus_b(h_drop, w_p, b_p, name="projected")
                        projected_nonlinearity = tf_utils.apply_nonlinearity(projected, params["nonlinearity"])
                    return projected_nonlinearity, projection_width

                # only use projection if we wanted to, and only apply middle dropout here if projection
                input_to_pred, proj_width = do_projection() if params["projection"] else (h_drop, total_output_width)
                input_to_pred_drop = tf.nn.dropout(input_to_pred, params["middle_dropout_keep_prob"]) \
                    if params["projection"] else input_to_pred

                # Final (unnormalized) scores and predictions
                with tf.name_scope("output" + block_name_suff):
                    w_o = tf_utils.initialize_weights([proj_width, params["num_classes"]], "w_o", init_type="xavier")
                    b_o = tf.get_variable("b_o", initializer=tf.constant(0.01, shape=[params["num_classes"]]))
                    self.l2_loss += tf.nn.l2_loss(w_o)
                    self.l2_loss += tf.nn.l2_loss(b_o)
                    scores = tf.nn.xw_plus_b(input_to_pred_drop, w_o, b_o, name="scores")
                    unflat_scores = tf.reshape(scores,
                                               tf.stack([params["batch_size"], max_seq_len, params["num_classes"]]))
                    block_unflat_scores.append(unflat_scores)

    return block_unflat_scores, h_concat_squeeze


def compute_loss(self, scores, scores_no_dropout, labels):
    loss = tf.constant(0.0)

    if params["CRF"]:
        zero_elements = tf.equal(self.sequence_lengths, tf.zeros_like(self.sequence_lengths))
        count_zeros_per_row = tf.reduce_sum(tf.to_int32(zero_elements), axis=1)
        flat_sequence_lengths = tf.add(tf.reduce_sum(self.sequence_lengths, 1),
                                       tf.scalar_mul(2, count_zeros_per_row))

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(scores, labels, flat_sequence_lengths,
                                                                              transition_params=self.transition_params)
        loss += tf.reduce_mean(-log_likelihood)
    else:
        if self.which_loss == "mean" or self.which_loss == "block":
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)
            masked_losses = tf.multiply(losses, self.input_mask)
            loss += tf.div(tf.reduce_sum(masked_losses), tf.reduce_sum(self.input_mask))
        elif self.which_loss == "sum":
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)
            masked_losses = tf.multiply(losses, self.input_mask)
            loss += tf.reduce_sum(masked_losses)
    loss += self.l2_penalty * self.l2_loss

    drop_loss = tf.nn.l2_loss(tf.subtract(scores, scores_no_dropout))
    loss += self.drop_penalty * drop_loss
    return loss


def model_fn(features, labels, mode, params):
    # For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read vocabs and inputs
    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    vocab_chars = tf.contrib.lookup.index_table_from_file(
        params['chars'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)
    variable = tf.get_variable(
        'chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    # char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
    #                                     training=training)

    # Char 1d convolution
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filters'], params['kernel_size'])

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


if __name__ == "__main__":
    # Params
    params = {
        'dim_chars': 100,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 15000,
        'filters': 50,
        'kernel_size': 3,
        'lstm_size': 100,
        "layers_map": "",
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz'))
    }
    with Path('results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    def fwords(name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))


    def ftags(name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))


    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('testa'), ftags('testa'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.contrib.estimator.stop_if_no_increase_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    # Write predictions to file
    def write_predictions(name):
        Path('results/score').mkdir(parents=True, exist_ok=True)
        with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), (_, _)), tags = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')


    for name in ['train', 'testa', 'testb']:
        write_predictions(name)
