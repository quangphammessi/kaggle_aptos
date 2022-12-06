import numpy as np
import pandas as pd

submission_csv = pd.read_csv('./data/sample_submission.csv')
NUMBER_OF_FOLD = 5
coef = [0.5, 1.5, 2.5, 3.5]


datafold = []

for i in range(NUMBER_OF_FOLD):
    datafold.append(pd.read_csv('./submission/submission_fold_%d.csv' %i))


result = []

for i in range(len(submission_csv)):
    res = 0
    for j in range(NUMBER_OF_FOLD):
        res += datafold[j]['diagnosis'][i]
    res /= NUMBER_OF_FOLD
    result.append(res)


for idx, value in enumerate(result):
    if value < coef[0]:
        result[idx] = 0
    elif value >= coef[0] and value < coef[1]:
        result[idx] = 1
    elif value >= coef[1] and value < coef[2]:
        result[idx] = 2
    elif value >= coef[2] and value < coef[3]:
        result[idx] = 3
    else:
        result[idx] = 4


submission_csv.diagnosis = np.array(result).astype(int)
submission_csv.to_csv("submission.csv", index=False)
print('Done!')