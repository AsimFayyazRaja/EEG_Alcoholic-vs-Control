import pickle

import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as aaa
from collections import Counter

with open('Scores/convsep2d_after_lstm_peakfeatures_predictions', 'rb') as fp:
    y_preds = pickle.load(fp)

with open('Scores/convsep2d_after_lstm_peakfeatures_reallabels', 'rb') as fp:
    y_reals = pickle.load(fp)


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(2):
    precision[i], recall[i], _ = precision_recall_curve(y_reals[:, i],
                                                        y_preds[:, i])
    average_precision[i] = average_precision_score(y_reals[:, i], y_preds[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_reals.ravel(),
    y_preds.ravel())
average_precision["micro"] = average_precision_score(y_reals, y_preds,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

from sklearn.utils.fixes import signature
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                 **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, Convsep2d after LSTM on full peaks features: AP={0:0.2f}'
    .format(average_precision["micro"]))

plt.show()

from sklearn.metrics import roc_auc_score
r=roc_auc_score(y_reals, y_preds)
print("ROC_AUC is: ", r)

'''
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_reals, y_preds)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

precision, recall, thresh = precision_recall_curve(y_reals, y_preds)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

plt.show()
'''

print(y_preds.shape)
print(y_reals.shape)
