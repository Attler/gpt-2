#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from spacy.lang.en import English

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

    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    df = pd.read_csv(input_dir)

    df = df[:10000]

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

        scores = []

        r1f_l = []
        r2f_l = []
        rlf_l = []

        for idx, row in df.iterrows():
            raw_text = row['story'] + ' TL;DR:'
            truth = row['headline']

            context_tokens = enc.encode(raw_text)

            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]

            text = enc.decode(out[0])
            doc = nlp(text)
            sentences = [sent.string.strip() for sent in doc.sents]
            if len(sentences[0]) > 10:
                text = sentences[0]

            score = utils.get_score(text, truth, verbose=False)
            scores.append(score)

            r1f = score[0]['rouge-1']['f']
            r2f = score[0]['rouge-2']['f']
            rlf = score[0]['rouge-l']['f']

            r1f_l.append(r1f)
            r2f_l.append(r2f)
            rlf_l.append(rlf)

            if(r1f > 0.65):
                print("=" * 40 + " INPUT " + str(idx) + " " + "=" * 40)
                print(raw_text)
                print("=" * 40 + " OUTPUT " + "=" * 40)
                print(text)
                print("=" * 40 + " TRUTH " + "=" * 40)
                print(truth)
                print('rouge-1 f: {} \t rouge-2 f: {} \t rouge-l f: {}\n'.format(r1f, r2f, rlf))

            if idx%100 == 0:
                r1f_m = np.mean(r1f_l)
                r2f_m = np.mean(r2f_l)
                rlf_m = np.mean(rlf_l)
                print('{{"metric": "rouge-1_f", "value": {}, "step": {}}}'.format(r1f_m, idx))
                print('{{"metric": "rouge-2_f", "value": {}, "step": {}}}'.format(r2f_m, idx))
                print('{{"metric": "rouge-l_f", "value": {}, "step": {}}}'.format(rlf_m, idx))


    df.to_csv("output.csv")


if __name__ == '__main__':
    fire.Fire(gen_headlines)

