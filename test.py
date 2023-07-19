# import transformers 
import torch 
import time 
import torch.nn as nn 

# out = torch.randint(0,10,size=(1,4,11))
# a = out[:,-1]
# print(out.size())
# print(a.size())
# print(out)
# print(a)
# out = torch.randint(0, 4, size=(2, 10))
# out_mask = (out != 2).unsqueeze(-2)
# print(out.size())
# print(out_mask.size())
# print(out)
# print(out_mask)
# out2 = out[:, :-1]
# print(out2.size())
# print(out2)
# out3 = out[:, 1:]
# print(out3.size())
# print(out3)

# print(time.time())
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
draw_loss(None, None)
