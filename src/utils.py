from rouge import Rouge


def get_score(hyp, ref, verbose=True):

    rouge = Rouge()
    score = rouge.get_scores(hyp, ref)
    if verbose:
        # metrics for floydhub
        print('{{"metric": "rouge-1_f", "value": {}}}'.format(score[0]['rouge-1']['f']))
        print('{{"metric": "rouge-1_p", "value": {}}}'.format(score[0]['rouge-1']['p']))
        print('{{"metric": "rouge-1_r", "value": {}}}'.format(score[0]['rouge-1']['r']))

        print('{{"metric": "rouge-2_f", "value": {}}}'.format(score[0]['rouge-2']['f']))
        print('{{"metric": "rouge-2_p", "value": {}}}'.format(score[0]['rouge-2']['p']))
        print('{{"metric": "rouge-2_r", "value": {}}}'.format(score[0]['rouge-2']['r']))

        print('{{"metric": "rouge-l_f", "value": {}}}'.format(score[0]['rouge-l']['f']))
        print('{{"metric": "rouge-l_p", "value": {}}}'.format(score[0]['rouge-l']['p']))
        print('{{"metric": "rouge-l_r", "value": {}}}'.format(score[0]['rouge-l']['r']))

    return score
