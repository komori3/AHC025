import os
import yaml
import math
from collections import defaultdict


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSIONS_DIR = os.path.join(ROOT_DIR, 'submissions')



def show_standings(dict_submission_to_total_score):
    max_length = len('submission')
    list_total_score_to_submission = []
    for submission, total_score in dict_submission_to_total_score.items():
        max_length = max(max_length, len(submission))
        list_total_score_to_submission.append((total_score, submission))

    list_total_score_to_submission.sort()

    space = max_length - len('submission') + 4
    print('submission' + (' ' * space) + 'score')
    print('-' * 50)
    for total_score, submission in list_total_score_to_submission:
        space = max_length - len(submission) + 4
        print(submission + (' ' * space) + str(total_score))

if __name__ == "__main__":

    submission_to_results = {}
    for tag in os.listdir(SUBMISSIONS_DIR):
        results_file = os.path.join(SUBMISSIONS_DIR, tag, 'results.yaml')
        with open(results_file) as f:
            submission_to_results[tag] = yaml.safe_load(f)

    seed_best = defaultdict(lambda: 1e20)
    for tag, results in submission_to_results.items():
        for result in results:
            seed_best[result['Seed']] = min(seed_best[result['Seed']], result['Score'])

    dict_submission_to_total_score = defaultdict(lambda: 0.0)
    for tag, results in submission_to_results.items():
        ctr = 0
        for result in results:
            ctr += 1
            if result['Score'] == -1: continue
            # dict_submission_to_total_score[tag] += seed_best[result['Seed']] / result['Score']
            # dict_submission_to_total_score[tag] += math.log(result['Score'])
            dict_submission_to_total_score[tag] += result['Score']
        dict_submission_to_total_score[tag] /= ctr

    show_standings(dict_submission_to_total_score)