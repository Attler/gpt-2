#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from rouge import Rouge
from tqdm import trange
import time
from spacy.lang.en import English

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
    input_dir='/floyd/input/news/test_enc.pkl',
    counter=0,
    prompt="briefly: "
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

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
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
        ckpt = tf.train.latest_checkpoint(os.path.join('checkpoint', 'run1'))
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))

        rouge = Rouge()
        context_enc = df['enc'].tolist()
        headlines = df['headline'].tolist()
        storys = df['story'].tolist()

        outputs = []

        for idx in range(len(df)):

            context_tokens = enc.encode(storys[idx] + prompt)

            outs = sess.run(output, feed_dict={context: [context_tokens]})
            out_txt = enc.decode(outs[0, len(context_tokens):])
            doc = nlp(out_txt)
            sentences = [sent.string.strip() for sent in doc.sents]
            out_txt = sentences[0]

            outputs.append(out_txt)

        with open('{}_output.txt'.format(time.strftime("%Y%m%d-%H%M%S")), 'w') as f:
            for item in outputs:
                f.write("%s\n" % item)

        score = rouge.get_scores(outputs, headlines, avg=True, ignore_empty=True)

        print('{{"metric": "rouge-1_f", "value": {}, "step": {}}}'.format(score['rouge-1']['f'], counter))
        print('{{"metric": "rouge-1_p", "value": {}, "step": {}}}'.format(score['rouge-1']['p'], counter))
        print('{{"metric": "rouge-1_r", "value": {}, "step": {}}}'.format(score['rouge-1']['r'], counter))

        print('{{"metric": "rouge-2_f", "value": {}, "step": {}}}'.format(score['rouge-2']['f'], counter))
        print('{{"metric": "rouge-2_p", "value": {}, "step": {}}}'.format(score['rouge-2']['p'], counter))
        print('{{"metric": "rouge-2_r", "value": {}, "step": {}}}'.format(score['rouge-2']['r'], counter))

        print('{{"metric": "rouge-l_f", "value": {}, "step": {}}}'.format(score['rouge-l']['f'], counter))
        print('{{"metric": "rouge-l_p", "value": {}, "step": {}}}'.format(score['rouge-l']['p'], counter))
        print('{{"metric": "rouge-l_r", "value": {}, "step": {}}}'.format(score['rouge-l']['r'], counter))

        print(prompt)

    return





if __name__ == '__main__':
    fire.Fire(gen_headlines)

