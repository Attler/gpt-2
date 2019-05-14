#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd

import model, sample, encoder, utils

def gen_headlines(
    model_name='345M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=25,
    temperature=1,
    top_k=2,
    input_dir='/floyd/input/news/abridged.csv'
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """

    df = pd.read_csv(input_dir)

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        for idx, row in df.iterrows():
            raw_text = row['story'] + ' TL;DR:'
            truth = row['headline']

            if idx < 1000 and idx%100 == 0:
                print("=" * 40 + " INPUT " + str(idx) + " " + "=" * 40)
                print(raw_text)

            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    if idx < 1000 and idx % 100 == 0:
                        print("=" * 40 + " OUTPUT " + "=" * 40)
                        print(text)
                        print("=" * 40 + " TRUTH " + "=" * 40)
                        print(truth)
                    df.loc[idx,'output'] = text
                    utils.get_score(text, truth)

    df.to_csv("output.csv")


if __name__ == '__main__':
    fire.Fire(gen_headlines)

