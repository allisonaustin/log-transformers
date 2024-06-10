import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import gc

def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0):
    line = [ln.split(",") for ln in line.split()]

    # filter the line/session shorter than 10
    if len(line) < min_len:
        return [], []

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)

    # if time duration exists in data
    if line.shape[1] == 2:
        tim = line[:,1].astype(float)
        line = line[:, 0]
        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze()
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)

    logkey_seqs = []
    time_seq = []
    for i in range(0, len(line), window_size):
        logkey_seqs.append(line[i:i + window_size])
        time_seq.append(tim[i:i + window_size])

    return logkey_seqs, time_seq


def generate_train_valid(data_path, window_size=20, adaptive_window=True,
                         sample_ratio=1, valid_size=0.1, output_path=None,
                         scale=None, scale_path=None, seq_len=None, min_len=0):

    # reading train data
    with open(data_path, 'r') as f:
        data_iter = f.readlines()

    num_session = int(len(data_iter) * sample_ratio)
    num_session += num_session % 2
    print('num session:', num_session)
    test_size = int(min(num_session, len(data_iter)) * valid_size)
    test_size += test_size % 2

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("="*40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    time_trainset = []
    time_validset = []
    session = 0
    # processing each session individually to ensure each session is entirely in train or validation set
    for line in tqdm(data_iter):
        if session >= num_session:
            break
        session += 1

        logkeys, times = fixed_window(line, window_size, adaptive_window, seq_len, min_len)
        logkey_seq_pairs += logkeys
        time_seq_pairs += times


    logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(logkey_seq_pairs,
                                                                                      time_seq_pairs,
                                                                                      test_size=test_size,
                                                                                      random_state=1234)
    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))

    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = [logkey_trainset[i] for i in train_sort_index]
    logkey_validset = [logkey_validset[i] for i in valid_sort_index]

    time_trainset = [time_trainset[i] for i in train_sort_index]
    time_validset = [time_validset[i] for i in valid_sort_index]

    print("="*40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("="*40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset

def generate_test(datapath, window_size, adaptive_window, seq_len, min_len):
    log_seqs = []
    tim_seqs = []
    with open(datapath, "r") as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            log_seq, tim_seq = fixed_window(line, window_size, adaptive_window=adaptive_window, seq_len=seq_len, min_len=min_len)
            if len(log_seq) == 0:
                continue

            log_seqs += log_seq
            tim_seqs += tim_seq

    test_len = list(map(len, log_seqs))
    test_sort_index = np.argsort(-1 * np.array(test_len))

    log_seqs = [log_seqs[i] for i in test_sort_index]
    tim_seqs = [tim_seqs[i] for i in test_sort_index]

    return log_seqs, tim_seqs