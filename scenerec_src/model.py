from torch import nn
import torch
import numpy as np
import time
import gc

output_each_running_time_state = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.ELU(),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ELU(),
        )

    def forward(self, x):
        return self.aggre(x)


class _UserModel(nn.Module):

    def __init__(self, emb_dim, item_emb, user_emb, pattern):
        super(_UserModel, self).__init__()
        self.emb_dim = emb_dim
        self.item_emb = item_emb
        self.user_emb = user_emb
        self.pattern = pattern

        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-10

    def forward(self, uids, u_item_pad):
        u_ngh = self.item_emb(u_item_pad)

        u_self_emb = self.user_emb(uids)
        u_ngh = torch.cat([u_self_emb.unsqueeze(1), u_ngh], dim=1)

        u_emb = self.aggre_items((
            torch.sum(u_ngh, 1)))


        return u_emb


class _CateModel(nn.Module):
    def __init__(self, emb_dim, cate_emb, scene_emb, cate_scene_pad, c_cate_pad, pattern):
        super(_CateModel, self).__init__()
        self.emb_dim = emb_dim
        self.cate_emb = cate_emb
        self.scene_emb = scene_emb
        self.cate_scene_pad = cate_scene_pad
        self.c_cate_pad = c_cate_pad
        self.pattern = pattern

        self.aggre_cates = _Aggregation(2 * self.emb_dim, self.emb_dim)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cids = torch.tensor(range(cate_scene_pad.size()[0])).to(self.device)
        self.eps = 1e-10

    def forward(self, cids):
        num_cate = self.c_cate_pad.size()[0]
        _list = list()
        for i in range(num_cate):
            _list.append(i)
        cids = torch.tensor(_list)
        cids = cids.type(torch.LongTensor).to(device)

        part_c_cate_pad = torch.cat([cids.unsqueeze(1), self.c_cate_pad], dim=1)

        c_ngh = self.cate_emb(part_c_cate_pad)
        mask_c = torch.where(part_c_cate_pad < self.cate_emb.weight.size()[0] - 1,
                             torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))

        c_cate_scene = self.cate_scene_pad[part_c_cate_pad]
        c_cate_scene_emb = self.scene_emb(c_cate_scene)
        right_c_cate_scene_emb = torch.sum(c_cate_scene_emb, 2)

        c_scene = self.cate_scene_pad
        c_scene_emb = self.scene_emb(c_scene)
        c_scene_emb_sum = torch.sum(c_scene_emb, 1)

        left_c_scene_emb = c_scene_emb_sum.unsqueeze(1).expand_as(right_c_cate_scene_emb) * mask_c.unsqueeze(
            2).expand_as(right_c_cate_scene_emb)

        left_c_scene_emb = left_c_scene_emb / (
                torch.sum(left_c_scene_emb ** 2, 2).unsqueeze(2).expand_as(left_c_scene_emb) + self.eps)
        right_c_cate_scene_emb = right_c_cate_scene_emb / (
                torch.sum(right_c_cate_scene_emb ** 2, 2).unsqueeze(2).expand_as(right_c_cate_scene_emb) + self.eps)
        scene_similarity = torch.sum((left_c_scene_emb * right_c_cate_scene_emb), 2)


        miu = torch.exp(scene_similarity) * mask_c
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)
        agg_cate_withAtt = torch.sum(miu.unsqueeze(2).expand_as(c_ngh) * c_ngh, 1)


        c_emb = self.aggre_cates(torch.cat([c_scene_emb_sum, agg_cate_withAtt], dim=1))


        return c_emb


class _ItemModel(nn.Module):
    def __init__(self, emb_dim, item_emb, user_emb, cate_emb, scene_emb, cate_scene_pad, c_cate_pad, i_cate_list, pattern):
        super(_ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.item_emb = item_emb
        self.user_emb = user_emb
        self.cate_emb = cate_emb
        self.scene_emb = scene_emb
        self.cate_scene_pad = cate_scene_pad
        self.i_cate_list = i_cate_list
        self.pattern = pattern

        self.cate_generator = _CateModel(emb_dim, cate_emb, scene_emb, cate_scene_pad, c_cate_pad, pattern=pattern)

        self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)

        self.aggre_items = _Aggregation(2 * self.emb_dim, self.emb_dim)

        self.aggre_users = _Aggregation(self.emb_dim, self.emb_dim)
        self.combine_mlp = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.ELU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ELU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ELU(),
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eps = 1e-10

    def forward(self, iids, i_item_pad, i_user_pad):
        i_self_emb = self.item_emb(iids)

        i_item_pad = torch.cat([iids.unsqueeze(1), i_item_pad], dim=1)
        i_nghItem = self.item_emb(i_item_pad)

        mask_i = torch.where(i_item_pad < self.item_emb.weight.size()[0] - 1, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        i_item_scene = self.cate_scene_pad[self.i_cate_list[i_item_pad]]
        i_item_scene_emb = self.scene_emb(i_item_scene)
        right_i_item_scene_emb = torch.sum(i_item_scene_emb, 2)
        i_scene = self.cate_scene_pad[self.i_cate_list[iids]]
        i_scene_emb = self.scene_emb(i_scene)
        i_scene_emb_sum = torch.sum(i_scene_emb, 1)
        left_c_scene_emb = i_scene_emb_sum.unsqueeze(1).expand_as(right_i_item_scene_emb) \
                           * mask_i.unsqueeze(2).expand_as(right_i_item_scene_emb)
        left_c_scene_emb = left_c_scene_emb / (
                torch.sum(left_c_scene_emb ** 2, 2).unsqueeze(2).expand_as(left_c_scene_emb) + self.eps)
        right_i_item_scene_emb = right_i_item_scene_emb / (
                    torch.sum(right_i_item_scene_emb ** 2, 2).unsqueeze(2).expand_as(right_i_item_scene_emb) + self.eps)
        scene_similarity = torch.sum((left_c_scene_emb * right_i_item_scene_emb), 2)
        miu = torch.exp(scene_similarity) * mask_i
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)
        agg_cate_withAtt = torch.sum(miu.unsqueeze(2).expand_as(i_nghItem) * i_nghItem, 1)


        i_cate = self.i_cate_list[iids]
        ready_cate_emb = self.cate_generator(cids=None)
        i_cate_emb = ready_cate_emb[i_cate]


        itemSpace_i_emb = self.aggre_items(torch.cat([i_cate_emb, agg_cate_withAtt], dim=1))

        i_nghUser = self.user_emb(i_user_pad)
        i_nghUser = torch.cat([i_self_emb.unsqueeze(1), i_nghUser], dim=1)
        userSpace_i_emb = self.aggre_users((torch.sum(i_nghUser, 1)) )


        i_emb = self.combine_mlp(torch.cat([userSpace_i_emb, itemSpace_i_emb], dim=1))

        return i_emb


class SceneRec(nn.Module):
    def __init__(self, num_users, num_cates, num_scenes, num_items, cate_scene_pad, c_cate_pad, i_cate_list,
                 emb_dim=64, pattern='normal'):
        super(SceneRec, self).__init__()
        self.num_users = num_users + 1
        self.num_cates = num_cates + 1
        self.num_scenes = num_scenes + 1
        self.num_items = num_items + 1
        self.cate_scene_pad = cate_scene_pad
        self.i_cate_list = i_cate_list
        self.emb_dim = emb_dim
        self.pattern = pattern

        self.cate_emb = nn.Embedding(self.num_cates, self.emb_dim, padding_idx=self.num_cates - 1)
        self.scene_emb = nn.Embedding(self.num_scenes, self.emb_dim, padding_idx=self.num_scenes - 1)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx=self.num_items - 1)
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim,
                                     padding_idx=self.num_users - 1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.item_model = _ItemModel(emb_dim=self.emb_dim, item_emb=self.item_emb, user_emb=self.user_emb,
                                     cate_emb=self.cate_emb, scene_emb=self.scene_emb,
                                     cate_scene_pad=self.cate_scene_pad, c_cate_pad=c_cate_pad,
                                     i_cate_list=self.i_cate_list, pattern=pattern)
        self.user_model = _UserModel(emb_dim=self.emb_dim, item_emb=self.item_emb, user_emb=self.user_emb, pattern=pattern)

        self.rate_pred = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.ELU(),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True),
            nn.ELU(),
            nn.Linear(self.emb_dim, 1),
        )
        self.eps = 1e-10

    def ready(self, uids=None, iids=None, u_item_pad=None, i_item_pad=None, i_user_pad=None, i_emb=None, u_emb=None,
              purpose='train'):
        if purpose == 'readyFORitem':
            i_emb = self.item_model(iids=iids, i_item_pad=i_item_pad, i_user_pad=i_user_pad)
            return i_emb
        elif purpose == 'readyFORuser':
            u_emb = self.user_model(uids=uids, u_item_pad=u_item_pad)
            return u_emb
        elif purpose == 'test':
            assert i_emb.size()[0] == u_emb.size()[0]
            r_ij = self.rate_pred(torch.cat([u_emb, i_emb], dim=1))
            return r_ij
        else:
            print('Error from SceneRec: the wrong state!')

    def forward(self, uids=None, iids=None, u_item_pad=None, i_item_pad=None, i_user_pad=None):
        u_emb = self.user_model(uids=uids, u_item_pad=u_item_pad)
        i_emb = self.item_model(iids=iids, i_item_pad=i_item_pad, i_user_pad=i_user_pad)
        r_ij = self.rate_pred(torch.cat([u_emb, i_emb], dim=1))

        return r_ij



class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.eps = 1e-10

    def forward(self, pos_scores, neg_scores):
        return torch.squeeze(torch.mean(-1 * torch.log(torch.sigmoid(pos_scores - neg_scores) + self.eps)))
