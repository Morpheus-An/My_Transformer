import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import math
def sequen_mask(X, valid_len, fill_value):
    
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = fill_value 
    return X  

def masked_softmax(X, valid_lens=None):
    if valid_lens == None:
        return F.softmax(X, dim=-1)
    shape = X.shape 
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, X.shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    X = sequen_mask(X.reshape(-1, shape[-1]), valid_lens, -1e6)
    return F.softmax(X.reshape(shape), dim=-1)


class MutiAtten(nn.Module):
    """多头注意力层, 当forward的参数只有x时输入x.shape = (batch_size, len, ndim) out.shape = (batch_size, len, ndim)
    否则，输入k,q,v(batch_size, len, dim) --> out (batch_size, len, ndim)"""
    def __init__(self, ndim, h, drop_v, is_cross=False):
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

        self.h = h 
        self.ndim = ndim 
        self.atten = None

    def forward(self, x, preinput, valid_lens):
        batch_size, len, ndim = preinput.size() 
        # self attention:
        if x == None:
            k, q, v = self.x2kqv(preinput).split(self.ndim, dim=2) # shape = (batch_size, len, ndim)

            # shape --> (batch_size, h, len, ndim // h)
            for c in (k, q, v):
                c = c.view(batch_size, len, self.h, -1).transpose(1, 2)
            atten_weight = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**(0.5)))

            atten_weight = masked_softmax(atten_weight, valid_lens)
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
            
            q = self.wq(preinput).view(batch_size, -1, self.h, ndim // self.h).transpose(1, 2)
            k = self.wk(x).view(batch_size, -1, self.h, ndim // self.h).transpose(1, 2)
            v = self.wv(x).view(batch_size, -1, self.h, ndim // self.h).transpose(1, 2)
            # shape = (batch_size, h, len, ndim // h)
            atten_weight = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))) # (batch_size, h, lenq, lenk)
            atten_weight = masked_softmax(atten_weight, valid_lens)
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
        self.attn = MutiAtten(ndim, h, drop_v, False)
        self.feedforward = fdfwd(ndim, ndim_hidden, drop_v)

        # self.layernorm0 = LayerNorm(ndim)
        self.layernorm1 = LayerNorm(ndim)
        self.layernorm2 = LayerNorm(ndim)
        self.dropout = nn.Dropout(drop_v)
    
    def forward(self, x, valid_len):
        # x = self.layernorm0(x) #! 为了de一个bug，加上一个层归一化
        x = x + self.dropout(self.attn(None, x, valid_len))
        x = self.layernorm1(x)
        x = x + self.dropout(self.feedforward(x))
        x = self.layernorm2(x)
        return x 

class decoder(nn.Module):
    """一个解码器块"""
    def __init__(self, ndim, h, drop_v, masked_len, ndim_hidden):
        super(decoder, self).__init__()
        self.selfattn = MutiAtten(ndim, h, drop_v, False)
        self.crossattn = MutiAtten(ndim, h, drop_v, True)
        self.feedforward = fdfwd(ndim, ndim_hidden, drop_v)
        self.dropout =nn.Dropout()
        
        # self.layernorm0 = LayerNorm(ndim) #!debug
        self.layernorm1 = LayerNorm(ndim)
        self.layernorm2 = LayerNorm(ndim)
        self.layernorm3 = LayerNorm(ndim)



    def forward(self, x, preinput, valid_len):
        # x = self.layernorm0(x) #! 同上
        preinput = preinput + self.dropout(self.selfattn(None, preinput, valid_len))
        preinput = self.layernorm1(preinput)
        preinput = preinput + self.dropout(self.crossattn(x, preinput, valid_len))
        preinput = self.layernorm2(preinput)
        preinput = preinput + self.dropout(self.feedforward(preinput))
        preinput = self.layernorm3(preinput)
        return preinput 

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
        self.dense = nn.Linear(ndim, vocab_tar)
        self.generator = Generator(ndim, vocab_tar)
    def encode(self, x, valid_lens):
        """接收的是尚未编码的向量"""
        # x.shape = (batch_size, len, ndim)
        for layer in self.prepros_inputlayer:
            x = layer(x)
        for layer in self.EncoderBlocks:
            x = layer(x, valid_lens)
        return x 

    def decode(self, x, preinput, valid_lens):
        """同上"""
        for layer in self.prepros_outlayer:
            preinput = layer(preinput)
        for layer in self.DecoderBlocks:
            preinput = layer(x, preinput, valid_lens)
        return preinput 
    
    def forward(self, x, preinput, enc_valid_lens, dec_valid_lens):
        """抛去generate的部分"""
        out =  self.decode(self.encode(x, enc_valid_lens), preinput, dec_valid_lens)
        return self.dense(out)

    def generate(self, x):
        """生成最终的概率"""
        return self.generator(x)

def inference_test():
    model = MYTransformer(512, 11, 11, 0.1, 8, 2048, 2, 12)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) # (1, 10)
    x = model.encode(src) # (1, 10, 512)
    preout = torch.zeros(1, 1).type_as(src)
    for i in range(9):
        out = model.decode(x, preout)  # (1, 1, 512) --> (1, 2, 512) --> ...
        prob = model.generate(out[:,-1]) # (1, 11) 
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        preout = torch.cat([preout, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    print("Example Untrained Model Prediction:", preout)
 

if __name__ == "__main__":

    def execute_inference_test(times):
        for _ in range(times):
            inference_test()
    execute_inference_test(10)


    

        




    
    

