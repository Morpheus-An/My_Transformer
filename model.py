import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import math
class MutiAtten(nn.Module):
    """多头注意力层, 当forward的参数只有x时输入x.shape = (batch_size, len, ndim) out.shape = (batch_size, len, ndim)
    否则，输入k,q,v(batch_size, len, dim) --> out (batch_size, len, ndim)"""
    def __init__(self, ndim, h, drop_v, masked_len=None, is_cross=False):
        super(MutiAtten, self).__init__()
        assert ndim % h == 0 
        # x.shape = (batch_size, len, ndim)
        if not is_cross:
            self.x2kqv = nn.Linear(ndim, 3 * ndim)  # out.shape = (batch_size, len, ndim)
        else:
            self.wq = nn.Linear(ndim, ndim)
            self.wv = nn.Linear(ndim, ndim)
            self.wk = nn.Linear(ndim, ndim)

        self.final_proj = nn.Linear(ndim, ndim)

        self.dropout = nn.Dropout(drop_v)
        # mask matrix(optional) #todo 初始化一个masked_len * masked_len 的掩码矩阵
        if masked_len != None:
            self.register_buffer("mask", torch.tril(torch.ones(masked_len, masked_len)).view(1, 1, masked_len, masked_len))

        self.h = h 
        self.ndim = ndim 
        self.masked_len = masked_len
        self.atten = None

    def forward(self, x, preinput=None):
        batch_size, len, ndim = x.size() 
        # self attention:
        if preinput == None:
            k, q, v = self.x2kqv(x).split(self.ndim, dim=2) # shape = (batch_size, len, ndim)

            # shape --> (batch_size, h, len, ndim // h)
            for c in (k, q, v):
                c = c.view(batch_size, len, self.h, -1).transpose(1, 2)
            atten_weight = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**(0.5)))
            if self.masked_len != None:
                atten_weight = atten_weight.masked_fill(self.mask[:,:,:len,:len] == 0, float('-inf'))
            atten_weight = F.softmax(atten_weight, dim=-1)
            atten_weight = self.dropout(atten_weight) # shape = (batch_size, h, len, len)

            out = atten_weight @ v # shape = (batch_size, h, len, ndim // h)
            self.attenn = atten_weight
            # concat:
            out = out.transpose(1, 2).contiguous().view(batch_size, len, ndim)

            # project:
            out = self.final_proj(out)
            return out  
        # todo cross attention:
        else:
            
            q = self.wq(x).view(batch_size, -1, self.h, ndim // self.h).transpose(1, 2)
            k = self.wk(preinput).view(batch_size, -1, self.h, ndim // self.h).transpose(1, 2)
            v = self.wv(preinput).view(batch_size, -1, self.h, ndim // self.h).transpose(1, 2)
            # shape = (batch_size, h, len, ndim // h)
            atten_weight = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))) # (batch_size, h, len, len)
            if self.masked_len != None:
                atten_weight = atten_weight.masked_fill(self.mask[:,:,:len,:len] == 0, float('-inf'))
            atten_weight = F.softmax(atten_weight, dim=-1)
            atten_weight = self.dropout(atten_weight)
            out = atten_weight @ v 
            self.atten = atten_weight 
            out = out.transpose(1, 2).contiguous().view(batch_size, len, ndim)
            out = self.final_proj(out)
            return out 
                    

class fdfwd(nn.Module):
    """feedforward层"""
    def __init__(self, ndim, ndim_hidden, drop_v=0.1):
        super(fdfwd, self).__init__()
        self.w1 = nn.Linear(ndim, ndim_hidden)
        self.w2 = nn.Linear(ndim_hidden, ndim)
        self.dropout = nn.Dropout(drop_v)
    
    def forward(self, x):
        out = self.w1(x).relu()
        out = self.dropout(out)
        out = self.w2(out)

        return out  
class wordEmbedding(nn.Module):
    """修改后的embedding层，就是将向量乘以sqrt(ndim)"""
    def __init__(self, ndim, vocab):
        super(wordEmbedding, self).__init__()
        self.embd = nn.Embedding(vocab, ndim)
        self.ndim = ndim 

    def forward(self, x):
        return self.embd(x) * math.sqrt(self.ndim)

class postionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, ndim, drop_v, maxlen=5000):
        super(postionalEncoding, self).__init__()
        PE = torch.zeros(maxlen, ndim)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, ndim, 2) * -(math.log(10000.0) / ndim)) 
        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)
        PE = PE.unsqueeze(0)
        self.register_buffer("PE", PE)
        self.dropout = nn.Dropout(drop_v)
    
    def forward(self, x):
        return self.dropout(x + self.PE[:,:x.size(1)].requires_grad_(False))

class LayerNorm(nn.Module):
    def __init__(self, ndim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(ndim))
        self.eps = eps
        self.b = nn.Parameter(torch.zeros(ndim))

    def forward(self, x):
        return self.a * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) + self.b

class encoder(nn.Module):
    """一个编码器块儿"""
    def __init__(self, drop_v, ndim, h, ndim_hidden):
        super(encoder, self).__init__()
        self.attn = MutiAtten(ndim, h, drop_v)
        self.feedforward = fdfwd(ndim, ndim_hidden, drop_v)

        self.layernorm0 = LayerNorm(ndim)
        self.layernorm1 = LayerNorm(ndim)
        self.layernorm2 = LayerNorm(ndim)
        self.dropout = nn.Dropout(drop_v)
    
    def forward(self, x):
        x = self.layernorm0(x) #! 为了de一个bug，加上一个层归一化
        x = x + self.dropout(self.attn(x))
        x = self.layernorm1(x)
        x = x + self.dropout(self.feedforward(x))
        x = self.layernorm2(x)
        return x 

class decoder(nn.Module):
    """一个解码器块"""
    def __init__(self, ndim, h, drop_v, masked_len, ndim_hidden):
        super(decoder, self).__init__()
        self.selfattn = MutiAtten(ndim, h, drop_v, masked_len)
        self.crossattn = MutiAtten(ndim, h, drop_v, None, True)
        self.feedforward = fdfwd(ndim, ndim_hidden, drop_v)
        self.dropout =nn.Dropout()
        
        self.layernorm0 = LayerNorm(ndim) #!debug
        self.layernorm1 = LayerNorm(ndim)
        self.layernorm2 = LayerNorm(ndim)
        self.layernorm3 = LayerNorm(ndim)



    def forward(self, x, preinput):
        x = self.layernorm0(x) #! 同上
        x = x + self.dropout(self.selfattn(x))
        x = self.layernorm1(x)
        x = x + self.dropout(self.crossattn(x, preinput))
        x = self.layernorm2(x)
        x = x + self.dropout(self.feedforward(x))
        x = self.layernorm3(x)
        return x 

class Generator(nn.Module):
    """最后的生成器部分"""
    def __init__(self, ndim, vocab_size):
        super(Generator, self).__init__()
        self.lin = nn.Linear(ndim, vocab_size)

    def forward(self, x):
        return F.softmax(self.lin(x), dim=-1)


class MYTransformer(nn.Module):
    def __init__(self, ndim, vocab_src, vocab_tar, drop_v, h, ndim_hidden, num_layers, masked_len):
        super(MYTransformer, self).__init__()
        self.prepros_inputlayer = nn.Sequential(wordEmbedding(ndim, vocab_src),postionalEncoding(ndim, drop_v)) # (batch_size, len, vocab_size) --> (batch_size, len, ndim)
        self.prepros_outlayer = nn.Sequential(wordEmbedding(ndim, vocab_tar),postionalEncoding(ndim, drop_v)) # (batch_size, len, vocab_size) --> (batch_size, len, ndim)
        self.dropout = nn.Dropout(drop_v)
        self.EncoderBlocks = nn.ModuleList([encoder(drop_v, ndim, h, ndim_hidden) for _ in range(num_layers)])
        # self.layernorm = LayerNorm(ndim)
        self.DecoderBlocks = nn.ModuleList([decoder(ndim, h, drop_v, masked_len, ndim_hidden) for _ in range(num_layers)])
        self.generator = Generator(ndim, vocab_tar)
    def encode(self, x):
        """接收的是尚未编码的向量"""
        # x.shape = (batch_size, len, ndim)
        for layer in self.prepros_inputlayer:
            x = layer(x)
        for layer in self.EncoderBlocks:
            x = layer(x)
        return x 

    def decode(self, x, preinput):
        """同上"""
        for layer in self.prepros_outlayer:
            x = layer(x)
        for layer in self.DecoderBlocks:
            x = layer(x, preinput)
        return x 
    
    def forward(self, x, preinput):
        """抛去generate的部分"""
        return self.decode(self.encode(x), preinput)

    def generate(self, x):
        return self.generator(x)

def inference_test():
    model = MYTransformer(512, 11, 11, 0.1, 8, 2048, 2, 12)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    preinput = model.encode(src)
    preout = torch.zeros(1, 1).type_as(src)
    for i in range(9):
        out = model.decode(preout, preinput)
        prob = model.generate(out[:,-1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        preout = torch.cat([preout, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    print("Example Untrained Model Prediction:", preout)


    
if __name__ == "__main__":

    def execute_inference_test(times):
        for _ in range(times):
            inference_test()
    execute_inference_test(10)


    

        




    
    

