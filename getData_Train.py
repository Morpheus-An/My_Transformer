import os
import torch
import sys 
import numpy as np  
import collections
import torch.utils.data as data 
import torch.nn as nn 
import time 
from model import *

def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_rnn_scratch`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
def read_data(data_path):
    """载入“英语－法语”数据集"""
    with open(os.path.join(data_path), 'r',
             encoding='utf-8') as f:
        return f.read()
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

# # print(text[:80])
# #@save
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# len(src_vocab)
# #@save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

# #@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len
# #@save
def load_data(batch_size, num_steps, data_path, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data(data_path))
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    # import d2l 
    # data_iter = d2l.load_array(data_arrays, batch_size)
    dataset = data.TensorDataset(*data_arrays)
    data_iter = data.DataLoader(dataset, batch_size, shuffle=True)
    return data_iter, src_vocab, tgt_vocab


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    epoches = [epoch+1 for epoch in range(num_epochs)]
    loss_per_epoch = []
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss',
    #                  xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        # timer = d2l.Timer()
        time_start = time.time()
        # metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        Loss_ntoken = [0, 0]
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            # Y_hat, _ = net(X, dec_input, X_valid_len)
            Y_hat = net(X, dec_input)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            # grad_clipping(net, 1)

            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                # metric.add(l.sum(), num_tokens)
                Loss_ntoken[0] += l.sum() 
                Loss_ntoken[1] += num_tokens
            # print(f"training on {str(device)}")
        loss_per_epoch.append(Loss_ntoken[0] / Loss_ntoken[1])
        # if (epoch + 1) % 1 == 0:
            # animator.add(epoch + 1, (metric[0] / metric[1],))
        print(f'epoch {epoch} loss {Loss_ntoken[0] / Loss_ntoken[1]:.3f}, {Loss_ntoken[1] / (time.time()-time_start):.1f} '
            f'tokens/sec on {str(device)}')
        
    return epoches, loss_per_epoch
import matplotlib.pyplot as plt
def draw_loss(loss, epoch):
    # 训练过程中每个epoch的loss值，这里仅作示例，实际使用时需要替换为真实的数据
    # 例如：loss_values = [2.3, 1.8, 1.5, 1.3, ...]
    # loss_values = [2.3, 1.8, 1.5, 1.3, 1.1, 1.0, 0.9, 0.85, 0.8, 0.75]

    # # 创建一个表示epoch的列表
    # epochs = list(range(1, len(loss_values) + 1))

    # 绘制折线图
    plt.plot(epoch, loss, 'bo-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(sequence_mask(X, torch.tensor([1, 2])))
if __name__ == '__main__':
    train_iter, src_vocab, tgt_vocab = load_data(batch_size=2, num_steps=8, data_path=r'data\fra-eng\fra.txt')
    ndim = 32
    vocab_src = len(src_vocab)
    vocab_tar = len(tgt_vocab)
    drop_v = 0.1 
    h =4
    ndim_hidden = 32 
    num_layers = 2 
    masked_len = max(vocab_src, vocab_tar)
    model = MYTransformer(ndim, vocab_src, vocab_tar, drop_v, h, ndim_hidden, num_layers, masked_len)
    lr, num_epochs, device = 0.001, 300, torch.device('cuda')
    train_seq2seq(model, train_iter, lr, num_epochs, tgt_vocab, device)
