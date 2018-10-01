import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

summary_path = './summaries/'
curves_filename = 'learning_curves.pkl'

base_len = 5
max_accs = []
count = 0
for folder in sorted(os.listdir(summary_path)):
    if 'RNN' not in folder:
        continue

    count += 1
    path = summary_path + folder + '/' + curves_filename

    with open(path, 'rb') as fd:
        loss, acc = pickle.load(fd)
    max_accs.append((base_len*(count), max(acc)))
rnn_max_accs = np.array(max_accs)

max_accs = []
count = 0
for folder in sorted(os.listdir(summary_path)):
    if 'LSTM' not in folder:
        continue

    count += 1
    path = summary_path + folder + '/' + curves_filename

    with open(path, 'rb') as fd:
        loss, acc = pickle.load(fd)
    max_accs.append((base_len*(count), max(acc)))
lstm_max_accs = np.array(max_accs)

font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

plt.plot(rnn_max_accs[:, 0], rnn_max_accs[:, 1])
plt.plot(lstm_max_accs[:, 0], lstm_max_accs[:, 1])
plt.xticks(lstm_max_accs[:, 0])
plt.legend(("RNN", "LSTM"))
plt.ylabel("best accuracy")
plt.xlabel("input length")
plt.title("Best accracies LSTM vs RNN")
print("done")



