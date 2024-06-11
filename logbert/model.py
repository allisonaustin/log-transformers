import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
'''
LogBERT
Source: https://github.com/HelenGuohx/logbert
'''
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
class AdditiveAttention(nn.Module):
    """
    Compute 'Bahdanau attention'
    https://tomekkorbak.com/2020/06/26/implementing-attention-in-pytorch/
    """
    def __init__(self, d_k, attn_heads):
        super().__init__()
        self.W1 = nn.Linear(d_k, d_k)
        self.W2 = nn.Linear(d_k, d_k) 
        self.v = torch.nn.Parameter(
            torch.FloatTensor(d_k).uniform_(-0.1, 0.1))

    def forward(self, query, key, value, mask=None, dropout=None):
        weights = self.W1(query) + self.W2(value)
        # if dropout is not None:
        #     weights = dropout(weights)
        p_attn = torch.tanh(weights)
        return p_attn * self.v, p_attn

class HierarchicalAttention(nn.Module):
    """
    'Hierarchical Attention Network' (HAN)
    https://buomsoo-kim.github.io/attention/2020/03/26/Attention-mechanism-16.md/
    """
    def __init__(self, batch_size, d_model, h):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.word_attn = nn.Linear(d_model, d_model)
        self.u_w = nn.Linear(d_model, batch_size, bias=False)
        self.sent_attn = nn.Linear(d_model, d_model, bias=False)
        self.u_s = nn.Linear(d_model, 1, bias=False)
        self.dense_out = nn.Linear(d_model, h)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # (1) Apply word-level attention
        wordenc_out = torch.zeros((batch_size, seq_len, x.size(-1))).to(x.device)
        for i in range(batch_size):
            # (2) Compute word-level attention weights
            u_word = torch.tanh(self.word_attn(x[i]))
            # word_weights = F.softmax(self.u_w(u_word), dim=0)
            word_weights = F.softmax(self.u_w(u_word), dim=1).transpose(0, 1).unsqueeze(-1)
            # (3) Compute sentence-level representation
            sent_summ_vec = torch.sum(x[i] * word_weights, dim=0)
            wordenc_out[i] = sent_summ_vec

        # (4) Apply sentence-level attention
        u_sent = torch.tanh(self.sent_attn(wordenc_out))
        sent_weights = F.softmax(self.u_s(u_sent), dim=0)

        x = wordenc_out * sent_weights
        x = x.view(batch_size, seq_len, self.h * self.d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k) 
        return self.output_linear(x)
    
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, additive=False):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.additive = additive

        if additive: 
            self.attention = AdditiveAttention(d_k=self.d_k, attn_heads=h) 
        else: 
            self.attention = Attention()

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
                
        return self.output_linear(x)
    

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        self.time_embed = nn.Linear(1, embed_size)

    def forward(self, time_interval):
        return self.time_embed(time_interval)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1, is_logkey=True, is_time=False):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=max_len)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.time_embed = TimeEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.is_logkey = is_logkey
        self.is_time = is_time

    def forward(self, sequence, segment_label=None, time_info=None):
        x = self.position(sequence)
        # if self.is_logkey:
        x = x + self.token(sequence)
        if segment_label is not None:
            x = x + self.segment(segment_label)
        if self.is_time:
            x = x + self.time_embed(time_info)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, attn):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        :param attn: attention mechanism
        """

        super().__init__() 
        if attn == 'additive': 
            self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, additive=True)
        elif attn == 'hierarchical':
            self.attention = HierarchicalAttention(batch_size=32, d_model=hidden, h=attn_heads)
        else:
            self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, additive=False)
        
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT(nn.Module):
    def __init__(self, vocab_size, max_len=512, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, is_logkey=True, is_time=False, attn=None):
        """
        :param vocab_size: total vocabulary size
        :param hidden: hidden size of transformer
        :param n_layers: number of transformer layers
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, max_len=512, dropout=dropout, is_logkey=is_logkey, is_time=is_time)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, attn) for _ in range(n_layers)])


    def forward(self, x, segment_label=None, time_info=None):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_label, time_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x