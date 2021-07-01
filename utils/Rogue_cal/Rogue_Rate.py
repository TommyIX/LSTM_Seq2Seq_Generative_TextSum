from rouge import Rouge
import numpy as np
import time

path_true_headline = 'txt/true_headline_daily.txt'
path_pred_headline = 'txt/model_daily.txt'

t1 = time.time()

true_headline = []
pred_headline = []
scores = []

with open(path_true_headline) as f:
    line = f.readline()
    while line:
        true_headline.append(line)
        line = f.readline()

with open(path_pred_headline) as f:
    line = f.readline()
    while line:
        pred_headline.append(line)
        line = f.readline()

rouge = Rouge()

for i in range(len(true_headline)):
    score = rouge.get_scores(pred_headline[i], true_headline[i])
    r1f = score[0]['rouge-1']['f']
    r2f = score[0]['rouge-2']['f']
    rlf = score[0]['rouge-l']['f']
    scores.append([r1f, r2f, rlf])

scores = np.array(scores)
scores = np.mean(scores, 0)
print('[rouge-1 rouge-2 rouge-l] = ' + str(scores))

t2 = time.time()

print('Running time: %s Seconds' % (t2 - t1))

end = None