import transformers 
import torch 
import torch.nn as nn 

# out = torch.randint(0,10,size=(1,4,11))
# a = out[:,-1]
# print(out.size())
# print(a.size())
# print(out)
# print(a)
out = torch.randint(0, 4, size=(2, 10))
out_mask = (out != 2).unsqueeze(-2)
print(out.size())
print(out_mask.size())
print(out)
print(out_mask)
out2 = out[:, :-1]
print(out2.size())
print(out2)
out3 = out[:, 1:]
print(out3.size())
print(out3)
