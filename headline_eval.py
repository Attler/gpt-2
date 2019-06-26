#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from rouge import Rouge
import time


from src import model, sample, encoder

def gen_headlines(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=25,
    temperature=1,
    top_k=2,
    top_p=0.0,
    input_dir='/floyd/input/news/test_enc.pkl'
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

    df = pd.read_pickle(input_dir)
    df = df[:1000]

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    with open(os.path.join('/floyd/input/old/models', model_name, 'encoder.json'), 'r') as f:
        encod = json.load(f)
    with open(os.path.join('/floyd/input/old/models', model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

    enc = encoder.Encoder(encoder=encod, bpe_merges=bpe_merges)

    hparams = model.default_hparams()
    with open(os.path.join('/floyd/input/old/models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    print('Testing model on ', input_dir)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('/floyd/input/old/checkpoint/run1'))
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(os.path.join('/floyd/input/old/models', model_name))
        saver.restore(sess, ckpt)

        rouge = Rouge()

        context_enc = df['enc'].tolist()
        headlines = df['headline'].tolist()

        outputs = []
        R1f = []
        R2f = []
        RLf = []

        for idx in range(len(df)):
            outs = sess.run(output, feed_dict={context: [context_enc[idx]]})

            out_txt = enc.decode(outs[0, len(context_enc[idx]):])
            print(out_txt)
            out_txt = " ".join(out_txt.split('<|endoftext|>'))
            out_txt = " ".join(out_txt.split())
            out_txt = " ".join(out_txt.split('.'))

            if len(out_txt) <2:
                out_txt = 'NONE'

            outputs.append(out_txt)

            score = rouge.get_scores(out_txt, headlines[idx])

            R1f.append(score[0]['rouge-1']['f'])
            R2f.append(score[0]['rouge-2']['f'])
            RLf.append(score[0]['rouge-l']['f'])

        df['output'] = outputs
        df['rouge-1_f'] = R1f
        df['rouge-2_f'] = R2f
        df['rouge-l_f'] = RLf

        df.to_csv('{}_output.csv'.format(time.strftime("%Y%m%d-%H%M%S")))


        with open('{}_output.txt'.format(time.strftime("%Y%m%d-%H%M%S")), 'w') as f:
            for item in outputs:
                f.write("%s\n" % item)


        try:
            score = rouge.get_scores(outputs, headlines, avg=True, ignore_empty=True)

            print('{{"metric": "rouge-1_f", "value": {}}}'.format(score['rouge-1']['f']))
            print('{{"metric": "rouge-1_p", "value": {}}}'.format(score['rouge-1']['p']))
            print('{{"metric": "rouge-1_r", "value": {}}}'.format(score['rouge-1']['r']))

            print('{{"metric": "rouge-2_f", "value": {}}}'.format(score['rouge-2']['f']))
            print('{{"metric": "rouge-2_p", "value": {}}}'.format(score['rouge-2']['p']))
            print('{{"metric": "rouge-2_r", "value": {}}}'.format(score['rouge-2']['r']))

            print('{{"metric": "rouge-l_f", "value": {}}}'.format(score['rouge-l']['f']))
            print('{{"metric": "rouge-l_p", "value": {}}}'.format(score['rouge-l']['p']))
            print('{{"metric": "rouge-l_r", "value": {}}}'.format(score['rouge-l']['r']))

        except ValueError:
            print('=' * 20, 'ValueError', '=' * 20)
            print(outputs)
            print("=" * 40)
            print(headlines)


    return


if __name__ == '__main__':
    fire.Fire(gen_headlines)

