from collections import Counter, OrderedDict
import torchtext
from torchtext.vocab import vocab as Vocab

class TorchVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, datapath, counter=None, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        
        with open(datapath, 'r') as f:
            texts = f.readlines()

        counter = Counter()
        for line in texts:
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()
            for word in words:
                counter[word] += 1

        specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"]
        for tok in specials:
            del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        self.vocab = Vocab(ordered_dict)
        self.vocab.insert_token("<pad>", 0)
        self.vocab.insert_token("<unk>", 1)
        self.vocab.insert_token("<eos>", 2)
        self.vocab.insert_token("<sos>", 3)
        self.vocab.insert_token("<mask>", 4)
        self.vocab.set_default_index(self.vocab["<unk>"])

    
    def get_vocab(self):
        return self.vocab