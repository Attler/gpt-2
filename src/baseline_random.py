

import pandas as pd
import numpy as np
from spacy.lang.en import English

import utils

def baseline_random(
    input_dir='/floyd/input/news/test.csv',

):

    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    df = pd.read_csv(input_dir)

    scores = []

    r1f_l = []
    r2f_l = []
    rlf_l = []

    for idx, row in df.iterrows():
        raw_text = row['story']
        truth = row['headline']

        text = raw_text[:100]

        doc = nlp(raw_text)
        sentences = [sent.string.strip() for sent in doc.sents]
        sentences = [sent for sent in sentences if len(sent) > 10]

        if len(sentences) > 0:
            text = np.random.choice(sentences)

        score = utils.get_score(text, truth, verbose=False)
        scores.append(score)

        r1f = score[0]['rouge-1']['f']
        r2f = score[0]['rouge-2']['f']
        rlf = score[0]['rouge-l']['f']

        r1f_l.append(r1f)
        r2f_l.append(r2f)
        rlf_l.append(rlf)

        if idx > 0 and idx % 100 == 0:
            r1f_m = np.mean(r1f_l[-100:])
            r2f_m = np.mean(r2f_l[-100:])
            rlf_m = np.mean(rlf_l[-100:])
            print('{{"metric": "rouge-1_f", "value": {}, "step": {}}}'.format(r1f_m, idx))
            print('{{"metric": "rouge-2_f", "value": {}, "step": {}}}'.format(r2f_m, idx))
            print('{{"metric": "rouge-l_f", "value": {}, "step": {}}}'.format(rlf_m, idx))

    r1f_m = np.mean(r1f_l)
    r2f_m = np.mean(r2f_l)
    rlf_m = np.mean(rlf_l)
    print('{{"metric": "rouge-1_f", "value": {}, "step": {}}}'.format(r1f_m, len(df)))
    print('{{"metric": "rouge-2_f", "value": {}, "step": {}}}'.format(r2f_m, len(df)))
    print('{{"metric": "rouge-l_f", "value": {}, "step": {}}}'.format(rlf_m, len(df)))

if __name__ == '__main__':
    baseline_random()