import pandas as pd
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import glob

training_path = '../train_split.csv'
eval_path = '../val_split.csv'
test_path = '../test_split.csv'

def get_classes_count(path: str) -> Tuple[int, int]:
    dataset = pd.read_csv(path)
    labels = dataset.labels.tolist()
    negatives, positives = labels.count(0), labels.count(1)
    return negatives, positives

def get_eval_results(path: str) -> Tuple[List, List]:
    dataset = pd.read_csv(path)
    return dataset['eval_loss'].tolist(), dataset['eval_acc'].tolist()

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

def build_loss_acc_plots(losses: Dict, accs: Dict, epochs: int, description: str) -> None:
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    n_epochs = [_ for _ in range(1, epochs + 1)]
    for name, loss_method in losses.items():
        ax[0].plot(n_epochs, loss_method, label=name)
    if epochs == 5:
        ax[0].set_xlabel('Rounds')
        ax[0].set_ylabel('Test Loss')
    else:
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Evaluation Loss')
    ax[0].legend(loc='best')

    for name, acc_method in accs.items():
        ax[1].plot(n_epochs, acc_method, label=name)
    if epochs == 5:
        ax[1].set_xlabel('Rounds')
        ax[1].set_ylabel('Test Accuracy')
    else:
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Evaluation Accuracy')
    ax[1].legend(loc='best')
    fig.savefig('../figures/{}.png'.format(description))

normal_results = '../results/normal/training_results_0.015380967408418655_0.8021108179419525.csv'
last_bald = '../results/bald/Round_5/training_results_0.007767189294099808_0.8021108179419525_query_100.csv'
last_max_entropy = '../results/max_entropy/Round_5/training_results_0.0075342413038015366_0.7783641160949868_query_100.csv'
last_mean_std = '../results/mean_std/Round_5/training_results_0.007299212273210287_0.46701846965699206_query_100.csv'

normal_eval_loss, normal_eval_acc = get_eval_results(normal_results)
bald_eval_loss, bald_eval_acc = get_eval_results(last_bald)
max_entropy_eval_loss, max_entropy_eval_acc = get_eval_results(last_max_entropy)
mean_std_eval_loss, mean_std_eval_acc = get_eval_results(last_mean_std)

losses = {'normal': normal_eval_loss, 'bald': bald_eval_loss, 'max_entropy': max_entropy_eval_loss,
          'mean_std': mean_std_eval_loss}

accs = {'normal': normal_eval_acc, 'bald': bald_eval_acc, 'max_entropy': max_entropy_eval_acc,
          'mean_std': mean_std_eval_acc}

build_loss_acc_plots(losses, accs, description='normal_vs_final_al_rounds')

def build_al_loss_acc_plots(losses: Dict, accs: Dict, description: str, epochs: int = 100) -> None:
    fig, ax = plt.subplots(nrows=len(list(losses.keys())), ncols=2, sharex=True, figsize=(9.5, 9.5))
    n_epochs = [_ for _ in range(1, epochs + 1)]
    
    query_dict = {index + 1: query for index, query in enumerate([115, 100, 90, 80, 70, 60, 50])}
    for round, losses_lists in losses.items():
        for name, method_loss in losses_lists.items():
            ax[int(round)-1][0].plot(n_epochs, method_loss, label=name)    
        if int(round) == len(list(losses.keys())):
            ax[int(round)-1][0].set_xlabel('Epochs')
        if epochs == 5:
            ax[int(round)-1][0].set_xlabel('Rounds')
            ax[int(round)-1][0].set_ylabel('Test Loss_{}'.format(query_dict[int(round)]))
        else:
            ax[int(round)-1][0].set_ylabel('Loss_{}'.format(query_dict[int(round)]))

    for round, accs_lists in accs.items():
        for name, method_acc in accs_lists.items():
            ax[int(round)-1][1].plot(n_epochs, method_acc, label=name)    
        if int(round) == len(list(losses.keys())):
            ax[int(round)-1][1].set_xlabel('Epochs')
        if epochs == 5:
            ax[int(round)-1][0].set_xlabel('Rounds')
            ax[int(round)-1][0].set_ylabel('Test Loss_{}'.format(query_dict[int(round)]))
        else:
            ax[int(round)-1][1].set_ylabel('Accuracy_{}'.format(query_dict[int(round)]))
    
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes][:1]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    fig.savefig('../figures/{}.png'.format(description))

al_methods = ['bald', 'mean_std', 'max_entropy']
losses, accs = {}, {}
for round in range(1, 6):
    round_data_loss, round_data_acc = [], []
    loss_dict, acc_dict = {}, {}
    for method in al_methods:
        path = '../results/{}/Round_{}/*_query_100.csv'.format(method, round)
        files = glob.glob(path)
        real_path = files[0]
        loss, acc = get_eval_results(real_path)
        loss_dict[method] = loss
        acc_dict[method] = acc
        splits = real_path.split('_')
        test_loss, test_acc = splits[-4], splits[-3]
        print(test_loss, test_acc)
    losses[str(round)] = loss_dict 
    accs[str(round)] = acc_dict

build_al_loss_acc_plots(losses, accs, description="al_rounds_performances")

tests_loss = {'bald': [], 'mean_std': [], 'max_entropy': []}
tests_accs = {'bald': [], 'mean_std': [], 'max_entropy': []}

for round in range(1, 6):
    for method in al_methods:
        path = '../results/{}/Round_{}/*_query_100_False.csv'.format(method, round)
        files = glob.glob(path)
        real_path = files[0]
        splits = real_path.split('_')
        test_loss, test_acc = splits[-4], splits[-3]
        tests_loss[method].append(float(test_loss))
        tests_accs[method].append(float(test_acc))

build_loss_acc_plots(tests_loss, tests_accs, 5, description='al_rounds_tests')

losses, accs = {}, {}
tests_loss_false = {'bald': [], 'mean_std': [], 'max_entropy': []}
tests_accs_false = {'bald': [], 'mean_std': [], 'max_entropy': []}

for round in range(1, 6):
    round_data_loss, round_data_acc = [], []
    loss_dict, acc_dict = {}, {}
    for method in al_methods:
        path = '../results/{}/Round_{}/*_query_100_False.csv'.format(method, round)
        files = glob.glob(path)
        real_path = files[0]
        loss, acc = get_eval_results(real_path)
        loss_dict[method] = loss
        acc_dict[method] = acc
        splits = real_path.split('_')
        test_loss, test_acc = splits[-5], splits[-4]
        tests_loss_false[method].append(float(test_loss))
        tests_accs_false[method].append(float(test_acc))
    losses[str(round)] = loss_dict 
    accs[str(round)] = acc_dict

build_al_loss_acc_plots(losses, accs, description="al_rounds_performances_false")
build_loss_acc_plots(tests_loss_false, tests_accs_false, 5, description='al_rounds_tests_false')

losses, accs = {}, {}
tests_loss = {'bald': [], 'mean_std': [], 'max_entropy': []}
tests_accs = {'bald': [], 'mean_std': [], 'max_entropy': []}

all_test_loss, all_test_accs = {}, {}
for index, query_size in enumerate([115, 100, 90, 80, 70, 60, 50]):
    round_data_loss, round_data_acc = [], []
    loss_dict, acc_dict = {}, {}
    for method in al_methods:
        path = '../results/{}/Round_1/*_query_{}.csv'.format(method, query_size)
        files = glob.glob(path)
        real_path = files[0]
        loss, acc = get_eval_results(real_path)
        loss_dict[method] = loss
        acc_dict[method] = acc
        splits = real_path.split('_')
        test_loss, test_acc = splits[-4], splits[-3]
        tests_loss[method].append(round(float(test_loss), 4))
        tests_accs[method].append(round(float(test_acc), 4))
    losses[str(index+1)] = loss_dict 
    accs[str(index+1)] = acc_dict

description_1 = "query_size_analyses"
build_al_loss_acc_plots(losses, accs, description=description_1)

def build_cm_plot(conf_matrix, description):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix {}'.format(description), fontsize=18)
    fig.savefig('../figures/{}.png'.format(description))