import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt

training_path = '../train_split.csv'
eval_path = '../val_split.csv'
test_path = '../test_split.csv'

def get_classes_count(path: str) -> Tuple[int, int]:
    dataset = pd.read_csv(path)
    labels = dataset.labels.tolist()
    negatives, positives = labels.count(0), labels.count(1)
    return negatives, positives

train_neg, train_pos = get_classes_count(training_path)
eval_neg, eval_pos = get_classes_count(eval_path)
test_neg, test_pos = get_classes_count(test_path)

negs = [train_neg, eval_neg, test_neg]
poss = [train_pos, eval_pos, test_pos]

frame = pd.DataFrame()
frame['split'] = ['Train', 'Eval', 'Test']
frame['Negative'] = negs
frame['Positive'] = poss

ax = frame.plot(x="split", y=["Negative", "Positive"], kind="bar", rot=0)
plt.title('Repartition of Labels across Splits')
plt.savefig('../figures/dataset_stats.png')
plt.show()