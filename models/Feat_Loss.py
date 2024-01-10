import torch
import torch.nn as nn


class Feat_Loss(nn.Module):
    def __init__(self):
        super(Feat_Loss, self).__init__()

    def forward(self, back_feat, feat_group, attn_group, device):
        top_k = len(attn_group)
        B, C, H, W = attn_group[0].shape
        batch_loss = 0

        sm_list = []
        for k in range(top_k):
            attn = attn_group[k]
            cur_max = attn.max(dim=1)[0]
            cur_flat = cur_max.reshape(B, H * W)
            cur_sm = cur_flat.softmax(dim=1)
            sm_list.append(cur_sm)

        for i in range(top_k):
            cur_sm = sm_list[i]
            for j in range(i + 1, top_k):
                com_sm = sm_list[j]
                batch_loss += (cur_sm * com_sm).sum()
        cal_num = top_k * (top_k - 1) // 2

        feat_loss = batch_loss / (cal_num * B)

        return feat_loss
