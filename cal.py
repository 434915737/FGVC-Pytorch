# cal.py
import torch
import torch.nn.functional as F

class CALoss(torch.nn.Module):
    def __init__(self):
        super(CALoss, self).__init__()

    def forward(self, attention_map):
        attention_map_mean = torch.mean(attention_map, dim=1, keepdim=True)
        return F.relu(attention_map - attention_map_mean)

def cal_attention(attention_map, cal_loss, cal_lambda):
    cal_map = cal_loss(attention_map)
    return attention_map + cal_lambda * cal_map
