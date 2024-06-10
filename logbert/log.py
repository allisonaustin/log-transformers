import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import defaultdict
import random
import numpy as np

from model import BERT

class LogDataset(Dataset):
    def __init__(self, log_corpus, time_corpus, vocab, seq_len, corpus_lines=None, encoding="utf-8", on_memory=True, predict_mode=False, mask_ratio=0.15):
        """
        :param corpus: log sessions/line
        :param vocab: log events collection including pad, ukn ...
        :param seq_len: max sequence length
        :param corpus_lines: number of log sessions
        :param encoding:
        :param on_memory:
        :param predict_mode: if predict
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.encoding = encoding

        self.predict_mode = predict_mode
        self.log_corpus = log_corpus
        self.time_corpus = time_corpus
        self.corpus_lines = len(log_corpus)

        self.mask_ratio = mask_ratio
        self.stoi = vocab.get_stoi()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        k, t = self.log_corpus[idx], self.time_corpus[idx]

        k_masked, k_label, t_masked, t_label = self.random_item(k, t)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        k = [self.vocab["<sos>"]] + k_masked
        k_label = [self.vocab["<pad>"]] + k_label
        # k_label = [self.vocab.sos_index] + k_label

        t = [0] + t_masked
        t_label = [self.vocab["<pad>"]] + t_label

        return k, k_label, t, t_label

    def random_item(self, k, t):
        tokens = list(k)
        output_label = []

        time_intervals = list(t)
        time_label = []

        for i, token in enumerate(tokens):
            time_int = time_intervals[i]
            prob = random.random()
            # replace 15% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                # raise AttributeError("no mask in visualization")

                if self.predict_mode:
                    tokens[i] = self.vocab['<mask>']
                    output_label.append(self.stoi.get(token, self.vocab['<unk>']))

                    time_label.append(time_int)
                    time_intervals[i] = 0
                    continue

                prob /= self.mask_ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab['<mask>']

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.stoi.get(token, self.vocab['<unk>'])

                output_label.append(self.stoi.get(token, self.vocab['<unk>']))

                time_intervals[i] = 0  # time mask value = 0
                time_label.append(time_int)

            else:
                tokens[i] = self.stoi.get(token, self.vocab['<unk>'])
                output_label.append(0)
                time_label.append(0)

        return tokens, output_label, time_intervals, time_label

    def collate_fn(self, batch, percentile=100, dynamical_pad=True):
        lens = [len(seq[0]) for seq in batch]

        # find the max len in each batch
        if dynamical_pad:
            # dynamical padding
            seq_len = int(np.percentile(lens, percentile))
            if self.seq_len is not None:
                seq_len = min(seq_len, self.seq_len)
        else:
            # fixed length padding
            seq_len = self.seq_len

        output = defaultdict(list)
        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]
            time_input = seq[2][:seq_len]
            time_label = seq[3][:seq_len]

            padding = [self.vocab['<pad>'] for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), time_input.extend(padding), time_label.extend(
                padding)

            time_input = np.array(time_input)[:, np.newaxis]
            output["bert_input"].append(bert_input)
            output["bert_label"].append(bert_label)
            output["time_input"].append(time_input)
            output["time_label"].append(time_label)

        output["bert_input"] = torch.tensor(output["bert_input"], dtype=torch.long)
        output["bert_label"] = torch.tensor(output["bert_label"], dtype=torch.long)
        output["time_input"] = torch.tensor(output["time_input"], dtype=torch.float)
        output["time_label"] = torch.tensor(output["time_label"], dtype=torch.float)

        return output

class BERTLog(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLogModel(self.bert.hidden, vocab_size)
        self.time_lm = TimeLogModel(self.bert.hidden)
        # self.fnn_cls = LinearCLS(self.bert.hidden)
        #self.cls_lm = LogClassifier(self.bert.hidden)
        self.result = {"logkey_output": None, "time_output": None, "cls_output": None, "cls_fnn_output": None}

    def forward(self, x, time_info):
        x = self.bert(x, time_info=time_info)

        self.result["logkey_output"] = self.mask_lm(x)
        self.result["time_output"] = self.time_lm(x)

        # self.result["cls_output"] = x.float().mean(axis=1) #x[:, 0]
        self.result["cls_output"] = x[:, 0]
        # self.result["cls_output"] = self.fnn_cls(x[:, 0])

        # print(self.result["cls_fnn_output"].shape)

        return self.result


class MaskedLogModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class TimeLogModel(nn.Module):
    def __init__(self, hidden, time_size=1):
        super().__init__()
        self.linear = nn.Linear(hidden, time_size)

    def forward(self, x):
        return self.linear(x)