#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""

"""

import os
import time
import datetime
import argparse
import pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join
import math
import gc
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

# from utils import collate_fn
from model import SceneRec
from model import BPRLoss
# from dataloader import GRDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_each_running_time_state = False



def load_fromMultiFile_full(base_dir, run_dir):

    cate_sceneNeighss = dict()
    with open(base_dir + 'cate_scene.txt') as f:
        print('2. Open file cate_scene.txt')
        for line in f:
            _list = line.strip().split('\t')
            if int(_list[0]) in cate_sceneNeighss:
                cate_sceneNeighss[int(_list[0])].append(int(_list[1]))
            else:
                cate_sceneNeighss[int(_list[0])] = [int(_list[1])]
        f.close()
    max_cateSceneNghNum = 0
    max_scene = 0
    for k, vs in cate_sceneNeighss.items():
        if len(vs) > max_cateSceneNghNum:
            max_cateSceneNghNum = len(vs)
        for v in vs:
            if v > max_scene:
                max_scene = v
    num_scene = max_scene + 1
    num_cate = len(cate_sceneNeighss)
    cate_scene_pad = torch.full((num_cate+1, max_cateSceneNghNum), num_scene)
    cate_scene_pad = cate_scene_pad.type(torch.LongTensor).to(device)
    for k, vs in cate_sceneNeighss.items():
        for i in range(len(vs)):
            cate_scene_pad[k][i] = cate_sceneNeighss[k][i]


    item_cateNeighs = dict()
    with open(base_dir + 'item_cate.txt') as f:
        print('1. Open file item_cate.txt')
        for line in f:
            _list = line.strip().split('\t')
            item_cateNeighs[int(_list[0])] = int(_list[1])
        f.close()
    i_cate_list = torch.empty(len(item_cateNeighs)+1)
    i_cate_list = i_cate_list.type(torch.LongTensor).to(device)
    for k, v in item_cateNeighs.items():
        i_cate_list[k] = v
    i_cate_list[len(item_cateNeighs)] = num_cate
    num_item = len(item_cateNeighs)


    item_itemNeighss_dict = dict()
    with open(base_dir + 'item_item.txt') as f:
        print('3. Open file item_item.txt')
        for line in f:
            _list = line.strip().split('\t')
            if int(_list[0]) in item_itemNeighss_dict:
                item_itemNeighss_dict[int(_list[0])].append(int(_list[1]))
            else:
                item_itemNeighss_dict[int(_list[0])] = [int(_list[1])]
        f.close()
    max_itemItemNghNum = 0
    for k, vs in item_itemNeighss_dict.items():
        if len(vs) > max_itemItemNghNum:
            max_itemItemNghNum = len(vs)
    i_item_pad = torch.full((num_item, max_itemItemNghNum), num_item)
    i_item_pad = i_item_pad.type(torch.LongTensor)
    for k, vs in item_itemNeighss_dict.items():
        for i in range(len(vs)):
            i_item_pad[k][i] = item_itemNeighss_dict[k][i]


    cate_cateNeighss = dict()
    with open(base_dir + 'cate_cate.txt') as f:
        print('4. Open file cate_cate.txt')
        for line in f:
            _list = line.strip().split('\t')
            if int(_list[0]) in cate_cateNeighss:
                cate_cateNeighss[int(_list[0])].append(int(_list[1]))
            else:
                cate_cateNeighss[int(_list[0])] = [int(_list[1])]
        f.close()
    max_cateCateNghNum = 0
    for k, vs in cate_cateNeighss.items():
        if len(vs) > max_cateCateNghNum:
            max_cateCateNghNum = len(vs)
    c_cate_pad = torch.full((num_cate+1, max_cateCateNghNum), num_cate)
    c_cate_pad = c_cate_pad.type(torch.LongTensor).to(device)
    for k, vs in cate_cateNeighss.items():
        for i in range(len(vs)):
            c_cate_pad[k][i] = cate_cateNeighss[k][i]


    user_itemNeighss_dict = dict()
    with open(run_dir + 'train_user_item.txt') as f:
        print('5. Open file train_user_item.txt')
        for line in f:
            _list = line.strip().split('\t')
            if int(_list[0]) in user_itemNeighss_dict:
                user_itemNeighss_dict[int(_list[0])].append(int(_list[1]))
            else:
                user_itemNeighss_dict[int(_list[0])] = [int(_list[1])]
        f.close()
    num_user = len(user_itemNeighss_dict)
    max_userItemNghNum = 0
    for k, vs in user_itemNeighss_dict.items():
        if len(vs) > max_userItemNghNum:
            max_userItemNghNum = len(vs)
    u_item_pad = torch.full((num_user, max_userItemNghNum), num_item)
    u_item_pad = u_item_pad.type(torch.LongTensor)
    for k, vs in user_itemNeighss_dict.items():
        for i in range(len(vs)):
            u_item_pad[k][i] = user_itemNeighss_dict[k][i]


    item_userNeighss_dict = dict()
    with open(run_dir + 'train_item_user.txt') as f:
        print('6. Open file train_item_user.txt')
        for line in f:
            _list = line.strip().split('\t')
            if int(_list[0]) in item_userNeighss_dict:
                item_userNeighss_dict[int(_list[0])].append(int(_list[1]))
            else:
                item_userNeighss_dict[int(_list[0])] = [int(_list[1])]
        f.close()
    max_itemUserNghNum = 0
    for k, vs in item_userNeighss_dict.items():
        if len(vs) > max_itemUserNghNum:
            max_itemUserNghNum = len(vs)
    i_user_pad = torch.full((num_item, max_itemUserNghNum), num_user)
    i_user_pad = i_user_pad.type(torch.LongTensor)
    for k, vs in item_userNeighss_dict.items():
        for i in range(len(vs)):
            i_user_pad[k][i] = item_userNeighss_dict[k][i]

    print('There should be 6 file')


    print('load training data')
    train_user = list()
    with open(run_dir + 'train_user.txt') as f:
        for line in f:
            _list = line.strip().split('\t')
            train_user.append(int(_list[0]))
        f.close()
    train_uids = torch.tensor(train_user)
    train_uids = train_uids.type(torch.LongTensor).to(device)
    train_posItem = list()
    with open(run_dir + 'train_posItem.txt') as f:
        for line in f:
            _list = line.strip().split('\t')
            train_posItem.append(int(_list[0]))
        f.close()
    train_pos_iids = torch.tensor(train_posItem)
    train_pos_iids = train_pos_iids.type(torch.LongTensor).to(device)
    train_negItem = list()
    with open(run_dir + 'train_negItem.txt') as f:
        for line in f:
            _list = line.strip().split('\t')
            train_negItem.append(int(_list[0]))
        f.close()
    train_neg_iids = torch.tensor(train_negItem)
    train_neg_iids = train_neg_iids.type(torch.LongTensor).to(device)
    assert len(train_user) == len(train_posItem) == len(train_negItem)


    print('Construct the related trained data')
    train_pos_u_item_pad = u_item_pad[train_uids].to(device)
    train_pos_i_item_pad = i_item_pad[train_pos_iids].to(device)
    train_pos_i_user_pad = i_user_pad[train_pos_iids].to(device)


    # train_neg
    train_neg_u_item_pad = u_item_pad[train_uids].to(device)
    train_neg_i_item_pad = i_item_pad[train_neg_iids].to(device)
    train_neg_i_user_pad = i_user_pad[train_neg_iids].to(device)

    # load valid
    valid_test_unique_uids = list()
    valid_test_unique_iids = list()

    print('load valid data')
    valid_user = list()
    with open(run_dir + 'valid_user.txt') as f:
        for line in f:
            _list = line.strip().split('\t')
            valid_user.append(int(_list[0]))
            if int(_list[0]) not in valid_test_unique_uids:
                valid_test_unique_uids.append(int(_list[0]))
        f.close()
    valid_uids = torch.tensor(valid_user)
    valid_uids = valid_uids.type(torch.LongTensor).to(device)
    valid_posItem = list()
    with open(run_dir + 'valid_posItem.txt') as f:
        for line in f:
            _list = line.strip().split('\t')
            valid_posItem.append(int(_list[0]))
            if int(_list[0]) not in valid_test_unique_iids:
                valid_test_unique_iids.append(int(_list[0]))
        f.close()
    valid_pos_iids = torch.tensor(valid_posItem)
    valid_pos_iids = valid_pos_iids.type(torch.LongTensor).to(device)
    valid_negItems = list()
    with open(run_dir + 'valid_negItems.txt') as f:
        for line in f:
            _list = line.strip().split('\t')
            temp_list = list()
            for tmp in _list:
                temp_list.append(int(tmp))
                if int(tmp) not in valid_test_unique_iids:
                    valid_test_unique_iids.append(int(tmp))
            valid_negItems.append(temp_list)
        f.close()
    valid_neg_iidss = torch.tensor(valid_negItems)
    valid_neg_iidss = valid_neg_iidss.type(torch.LongTensor).to(device)

    print('construct valid related data')

    valid_negPos_iids = torch.cat((valid_neg_iidss, valid_pos_iids.unsqueeze(1)), 1).to(device)


    print('load test test data')
    test_user = list()
    with open(run_dir + 'test_user.txt') as f:
        for line in f:
            _list = line.strip().split('\t')
            test_user.append(int(_list[0]))
            if int(_list[0]) not in valid_test_unique_uids:
                valid_test_unique_uids.append(int(_list[0]))
        f.close()
    test_uids = torch.tensor(test_user)
    test_uids = test_uids.type(torch.LongTensor).to(device)
    test_posItem = list()
    with open(run_dir + 'test_posItem.txt') as f:
        for line in f:
            _list = line.strip().split('\t')
            test_posItem.append(int(_list[0]))
            if int(_list[0]) not in valid_test_unique_iids:
                valid_test_unique_iids.append(int(_list[0]))
        f.close()
    test_pos_iids = torch.tensor(test_posItem)
    test_pos_iids = test_pos_iids.type(torch.LongTensor).to(device)
    test_negItems = list()
    with open(run_dir + 'test_negItems.txt') as f:
        for line in f:
            _list = line.strip().split('\t')
            temp_list = list()
            for tmp in _list:
                temp_list.append(int(tmp))
                if int(tmp) not in valid_test_unique_iids:
                    valid_test_unique_iids.append(int(tmp))
            test_negItems.append(temp_list)
        f.close()
    test_neg_iidss = torch.tensor(test_negItems).to(device)
    test_neg_iidss = test_neg_iidss.type(torch.LongTensor).to(device)

    print('construct test related data')
    test_negPos_iids = torch.cat((test_neg_iidss, test_pos_iids.unsqueeze(1)), 1).to(device)

    valid_test_unique_uids = torch.tensor(valid_test_unique_uids).to(device)
    valid_test_unique_u_item_pad = u_item_pad[valid_test_unique_uids].to(device)
    valid_test_unique_iids = torch.tensor(valid_test_unique_iids).to(device)
    valid_test_unique_i_item_pad = i_item_pad[valid_test_unique_iids].to(device)
    valid_test_unique_i_user_pad = i_user_pad[valid_test_unique_iids].to(device)

    return num_user, num_cate, num_scene, num_item, \
           i_cate_list, cate_scene_pad, c_cate_pad, item_itemNeighss_dict, user_itemNeighss_dict, item_userNeighss_dict, \
           train_uids, train_pos_iids, train_pos_u_item_pad, train_pos_i_item_pad, train_pos_i_user_pad, \
           train_neg_iids, train_neg_u_item_pad, train_neg_i_item_pad, train_neg_i_user_pad, \
           valid_uids, valid_negPos_iids, test_uids, test_negPos_iids, \
           valid_test_unique_uids, valid_test_unique_u_item_pad, \
           valid_test_unique_iids, valid_test_unique_i_item_pad, valid_test_unique_i_user_pad


def trainForEpoch_full(train_loader, model, optimizer, loss, epoch, num_epochs, out_dir, log_aggr=1):
    model.train()
    sum_epoch_loss = 0

    for i, (train_uids, train_pos_iids, train_pos_u_item_pad, train_pos_i_item_pad, train_pos_i_user_pad,
            train_neg_iids, train_neg_u_item_pad, train_neg_i_item_pad, train_neg_i_user_pad) in tqdm(
                enumerate(train_loader),total=len(train_loader)):

        train_uids = train_uids.to(device)
        train_pos_iids = train_pos_iids.to(device)
        train_pos_u_item_pad = train_pos_u_item_pad.to(device)
        train_pos_i_item_pad = train_pos_i_item_pad.to(device)
        train_pos_i_user_pad = train_pos_i_user_pad.to(device)
        train_neg_iids = train_neg_iids.to(device)
        train_neg_u_item_pad = train_neg_u_item_pad.to(device)
        train_neg_i_item_pad = train_neg_i_item_pad.to(device)
        train_neg_i_user_pad = train_neg_i_user_pad.to(device)

        optimizer.zero_grad()
        pos_scores = model(train_uids, train_pos_iids, train_pos_u_item_pad, train_pos_i_item_pad, train_pos_i_user_pad)
        neg_scores = model(train_uids, train_neg_iids, train_neg_u_item_pad, train_neg_i_item_pad, train_neg_i_user_pad)
        loss_val = loss(pos_scores, neg_scores)
        loss_val.backward()
        optimizer.step()
        loss_val = loss_val.item()
        sum_epoch_loss += loss_val

    with open(out_dir + 'train_loss.txt', 'a') as f:
        print(str(float(sum_epoch_loss / len(train_loader))), file=f)
    f.close()


def evaluate(scores, evaluation_interval):

    ndcgs = torch.zeros(scores.size()[0], scores.size()[1] // evaluation_interval)
    hrs = torch.zeros(scores.size()[0], scores.size()[1] // evaluation_interval)

    pos_index = scores.size()[1] - 1
    scores = scores.cpu()
    for i in range(scores.size()[0]):
        single_vector = scores[i].numpy()
        count = 0
        for top_n in range(evaluation_interval, scores.size()[1], evaluation_interval):
            ndcg = 0.0
            hr = 0.0
            arg_index = np.argsort(-single_vector)[:top_n]
            if pos_index in arg_index:
                ndcg += np.log(2.0) / np.log(arg_index.tolist().index(pos_index) + 2.0)
                hr += 1.0
            ndcgs[i][count] = ndcg
            hrs[i][count] = hr
            count += 1

    ndcg = torch.div(torch.sum(ndcgs, 0), torch.full([ndcgs.size()[1]], ndcgs.size()[0]))
    hr = torch.div(torch.sum(hrs, 0), torch.full([hrs.size()[1]], hrs.size()[0]))
    return ndcg, hr


def validate_full(valid_loader, model, existed_item_id2emb, existed_user_id2emb, evaluation_interval=5):
    model.eval()
    ndcgs = []
    hrs = []

    num_poch = 0
    with torch.no_grad():
        for i, (test_uids, test_negPos_iids) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            test_uids = test_uids.to(device)
            test_negPos_iids = test_negPos_iids.to(device)

            uids = test_uids.unsqueeze(1).expand_as(test_negPos_iids).contiguous().view(-1).to(device)
            iids = test_negPos_iids.view(-1).to(device)

            preds = model.ready(uids=None, iids=None, u_item_pad=None, i_item_pad=None, i_user_pad=None,
                                i_emb=existed_item_id2emb[iids], u_emb=existed_user_id2emb[uids], purpose='test').data

            preds = preds.view(test_negPos_iids.size())

            ndcg, hr = evaluate(preds, evaluation_interval)
            ndcgs.append(ndcg.numpy())
            hrs.append(hr.numpy())
            num_poch += 1

    ndcgs = torch.from_numpy(np.array(ndcgs))
    hrs = torch.from_numpy(np.array(hrs))
    ndcgs = torch.div(torch.sum(ndcgs, 0), torch.full([ndcgs.size()[1]], num_poch))
    hrs = torch.div(torch.sum(hrs, 0), torch.full([hrs.size()[1]], num_poch))

    return ndcgs, hrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='normal',
                        help="Pattern of model: 'normal', 'noitem', 'nosce', 'noatt'")  #
    parser.add_argument('--dataset', default='electronics',
                        help="Chose of dataset: 'baby_toy', 'electronics', 'fashion', 'food_drink'") #
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='the dimension of embedding')
    parser.add_argument('--epoch', type=int, default=30,
                        help='the number of epochs to train for')
    parser.add_argument('--lr',
                        type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_norm_lamda', type=float,
                        default=0.1, help='Weight of L2 norm model parameters')
    parser.add_argument('--alpha', type=float,
                        default=0.999, help='Basis of RMSprop')
    parser.add_argument('--momentum', type=float,
                        default=0.1, help='Momentum of RMSprop')
    parser.add_argument('--lr_dc', type=float,
                        default=0.1, help='learning rate decay rate')
    parser.add_argument('--lr_dc_step',
                        type=int, default=30,
                        help='the number of steps after which the learning rate decay')
    args = parser.parse_args()
    print(args)
    # assert args.pattern == 'normal' or args.pattern == 'noitem' or args.pattern == 'nosce' or args.pattern == 'noatt'

    print('Loading data...')
    print('Now the model is 1: full data adjacency list')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())

    dataSetName = args.dataset
    existed_in_dir = '../in_' + dataSetName + '/'
    static_in_dir = '../../pub_' + dataSetName + '/static/'
    run_in_dir = '../../pub_' + dataSetName + '/run/'
    now = datetime.datetime.now()
    now = int(time.time())
    timeArray = time.localtime(now)
    StyleTime = time.strftime("%Y-%m-%d_%H-%M-%S", timeArray)
    argsStateStr = '%s_EmbDim-%s_Batch-%s_Epoch-%s_LR-%s_L2Lamda-%s_alpha-%s_momentum-%s_LRDC-%s_LRDCStep-%s' % \
                   (str(args.pattern), str(args.embed_dim), str(args.batch_size), str(args.epoch), str(args.lr),
                    str(args.l2_norm_lamda), str(args.alpha), str(args.momentum), str(args.lr_dc), str(args.lr_dc_step))
    super_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    abs_dir = super_dir + '/out_' + dataSetName + '_' + argsStateStr + '_' + str(StyleTime)
    print('Outdir: ' + abs_dir)
    os.mkdir(abs_dir)
    assert os.path.exists('../out_' + dataSetName + '_' + argsStateStr + '_' + str(StyleTime))
    out_dir = '../out_' + dataSetName + '_' + argsStateStr + '_' + str(StyleTime) + '/'

    in_file = Path(existed_in_dir + 'save_term_in_data.pickle')
    if in_file.exists():
        with open(existed_in_dir + 'save_term_in_data.pickle', 'rb') as f:
            num_user, num_cate, num_scene, num_item, \
            i_cate_list, cate_scene_pad, c_cate_pad, item_itemNeighss_dict, user_itemNeighss_dict, item_userNeighss_dict, \
            train_uids, train_pos_iids, train_pos_u_item_pad, train_pos_i_item_pad, train_pos_i_user_pad, \
            train_neg_iids, train_neg_u_item_pad, train_neg_i_item_pad, train_neg_i_user_pad, \
            valid_uids, valid_negPos_iids, test_uids, test_negPos_iids, \
            valid_test_unique_uids, valid_test_unique_u_item_pad, \
            valid_test_unique_iids, valid_test_unique_i_item_pad, valid_test_unique_i_user_pad = pickle.load(f)
            f.close()
        print('The input data has been load from the pickle file')
    else:
        num_user, num_cate, num_scene, num_item, \
        i_cate_list, cate_scene_pad, c_cate_pad, item_itemNeighss_dict, user_itemNeighss_dict, item_userNeighss_dict, \
        train_uids, train_pos_iids, train_pos_u_item_pad, train_pos_i_item_pad, train_pos_i_user_pad, \
        train_neg_iids, train_neg_u_item_pad, train_neg_i_item_pad, train_neg_i_user_pad, \
        valid_uids, valid_negPos_iids, test_uids, test_negPos_iids, \
        valid_test_unique_uids, valid_test_unique_u_item_pad, \
        valid_test_unique_iids, valid_test_unique_i_item_pad, valid_test_unique_i_user_pad = load_fromMultiFile_full(
            static_in_dir, run_in_dir)

        with open(existed_in_dir + 'save_term_in_data.pickle', 'wb') as f:
            pickle.dump([num_user, num_cate, num_scene, num_item,
                         i_cate_list, cate_scene_pad, c_cate_pad, item_itemNeighss_dict, user_itemNeighss_dict,
                         item_userNeighss_dict,
                         train_uids, train_pos_iids, train_pos_u_item_pad, train_pos_i_item_pad, train_pos_i_user_pad,
                         train_neg_iids, train_neg_u_item_pad, train_neg_i_item_pad, train_neg_i_user_pad,
                         valid_uids, valid_negPos_iids, test_uids, test_negPos_iids,
                         valid_test_unique_uids, valid_test_unique_u_item_pad,
                         valid_test_unique_iids, valid_test_unique_i_item_pad, valid_test_unique_i_user_pad], f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        print('The input data has been dump in the pickle file')

    print('Date has load. user:%d, cate:%d, scene:%d, item:%d' % (num_user, num_cate, num_scene, num_item))


    print('train for pos: train_uids:' + str(train_uids.size()) + ', train_pos_iids:' + str(
        train_pos_iids.size()) + ', train_pos_u_item_pad:' + str(
        train_pos_u_item_pad.size()) + ', train_pos_i_item_pad:' + str(
        train_pos_i_item_pad.size()) + ', train_pos_i_user_pad:' + str(train_pos_i_user_pad.size()))
    print('train for pos: train_uids:' + str(train_uids.size()) + ', train_neg_iids:' + str(
        train_neg_iids.size()) + ', train_neg_u_item_pad:' + str(
        train_neg_u_item_pad.size()) + ', train_neg_i_item_pad:' + str(
        train_neg_i_item_pad.size()) + ', train_neg_i_user_pad:' + str(train_neg_i_user_pad.size()))
    print('valid: valid_uids:' + str(valid_uids.size()) + ', valid_negPos_iids:' + str(
        valid_negPos_iids.size()))
    print('test: test_uids:' + str(test_uids.size()) + ', test_negPos_iids:' + str(
        test_negPos_iids.size()))
    print('ready for valid_test: valid_test_unique_uids: ' + str(
        valid_test_unique_uids.size()) + 'valid_test_unique_u_item_pad: ' + str(
        valid_test_unique_u_item_pad.size()))
    print('ready for valid_test: valid_test_unique_iids: ' + str(
        valid_test_unique_iids.size()) + 'valid_test_unique_i_item_pad: ' + str(
        valid_test_unique_i_item_pad.size()) + 'valid_test_unique_i_user_pad: ' + str(
        valid_test_unique_i_user_pad.size()))


    trainset = torch.utils.data.TensorDataset(train_uids, train_pos_iids, train_pos_u_item_pad,
                                              train_pos_i_item_pad, train_pos_i_user_pad, train_neg_iids,
                                              train_neg_u_item_pad, train_neg_i_item_pad, train_neg_i_user_pad)
    validset = torch.utils.data.TensorDataset(valid_uids, valid_negPos_iids)
    testset = torch.utils.data.TensorDataset(test_uids, test_negPos_iids)

    readyFORvalidTest_userset = torch.utils.data.TensorDataset(valid_test_unique_uids, valid_test_unique_u_item_pad)
    readyFORvalidTest_itemset = torch.utils.data.TensorDataset(valid_test_unique_iids, valid_test_unique_i_item_pad,
                                                               valid_test_unique_i_user_pad)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    readyFORvalidTest_userloader = torch.utils.data.DataLoader(readyFORvalidTest_userset,
                                                               batch_size=args.batch_size, shuffle=True)
    readyFORvalidTest_itemloader = torch.utils.data.DataLoader(readyFORvalidTest_itemset,
                                                               batch_size=args.batch_size, shuffle=True)
    print('Dataset has proprecessed, including training, valid, test.')


    print('Init the model.')
    model = SceneRec(num_users=num_user, num_cates=num_cate, num_scenes=num_scene, num_items=num_item,
                     cate_scene_pad=cate_scene_pad, c_cate_pad=c_cate_pad, i_cate_list=i_cate_list,
                     emb_dim=args.embed_dim, pattern=args.pattern).to(device)


    existed_item_id2emb = torch.rand(num_item, args.embed_dim).to(device)
    existed_user_id2emb = torch.rand(num_user, args.embed_dim).to(device)


    optimizer = optim.RMSprop(model.parameters(), args.lr, weight_decay=args.l2_norm_lamda, alpha=args.alpha, momentum=args.momentum)
    loss = BPRLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)


    for epoch in tqdm(range(args.epoch)):
        scheduler.step(epoch=epoch)
        with open(out_dir + 'train_loss.txt', 'a') as f:
            print(str(epoch) + '\t', file=f, end='')
        f.close()
        trainForEpoch_full(train_loader, model, optimizer, loss, epoch, args.epoch, out_dir, log_aggr=10)

        model.eval()
        for i, (valid_test_unique_uids, valid_test_unique_u_item_pad) in tqdm(
                enumerate(readyFORvalidTest_userloader),
                total=len(readyFORvalidTest_userloader)):
            u_emb = model.ready(uids=valid_test_unique_uids, iids=None, u_item_pad=valid_test_unique_u_item_pad,
                                i_item_pad=None,
                                i_user_pad=None, i_emb=None, u_emb=None, purpose='readyFORuser').data
            for j in range(valid_test_unique_uids.size()[0]):
                existed_user_id2emb[valid_test_unique_uids[j]] = u_emb[j]

        # item
        for i, (valid_test_unique_iids, valid_test_unique_i_item_pad, valid_test_unique_i_user_pad) in tqdm(
                enumerate(readyFORvalidTest_itemloader),
                total=len(readyFORvalidTest_itemloader)):
            i_emb = model.ready(uids=None, iids=valid_test_unique_iids, u_item_pad=None,
                                i_item_pad=valid_test_unique_i_item_pad,
                                i_user_pad=valid_test_unique_i_user_pad, i_emb=None, u_emb=None,
                                purpose='readyFORitem').data
            for j in range(valid_test_unique_iids.size()[0]):
                existed_item_id2emb[valid_test_unique_iids[j]] = i_emb[j]


        with open(out_dir + 'user_and_item_embs.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([existed_item_id2emb, existed_user_id2emb], f)
        f.close()

        print('existed_item_id2emb.size: ' + str(existed_item_id2emb.size()))
        print('existed_user_id2emb.size: ' + str(existed_user_id2emb.size()))
        result1 = np.array(existed_item_id2emb.cpu())
        np.savetxt(out_dir + 'existed_item_id2emb.txt', result1)
        result2 = np.array(existed_user_id2emb.cpu())
        np.savetxt(out_dir + 'existed_user_id2emb.txt', result2)

        ndcgs, hrs = validate_full(valid_loader, model, existed_item_id2emb, existed_user_id2emb, evaluation_interval=5)
        with open(out_dir + 'last_ndcg_hr.txt', 'w') as f:
            for i in range(ndcgs.size()[0]):
                print(str(float(ndcgs[i])) + '\t' + str(float(hrs[i])), file=f)
            f.close()


        ndcg = float(torch.div(torch.sum(ndcgs, 0), torch.tensor(ndcgs.size()[0])))
        hr = float(torch.div(torch.sum(hrs, 0), torch.tensor(hrs.size()[0])))

        ckpt_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, out_dir + 'latest_checkpoint.pth.tar')
        print('the model para has been stored')


        if epoch == 0:
            best_ndcg = ndcg
            torch.save(ckpt_dict, out_dir + 'best_checkpoint.pth.tar')
        elif ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(ckpt_dict, out_dir + 'best_checkpoint.pth.tar')

        print('Epoch {} validation: ndcg: {:.4f}, hr: {:.4f}, Best ndcg: {:.4f}'.format(epoch, ndcg, hr, best_ndcg))


    print('Load checkpoint and testing...')
    ckpt = torch.load(out_dir + 'best_checkpoint.pth.tar')
    model.load_state_dict(ckpt['state_dict'])

    print('Ready for test')
    model.eval()
    for i, (valid_test_unique_uids, valid_test_unique_u_item_pad) in tqdm(
            enumerate(readyFORvalidTest_userloader),
            total=len(readyFORvalidTest_userloader)):
        u_emb = model.ready(uids=valid_test_unique_uids, iids=None, u_item_pad=valid_test_unique_u_item_pad,
                            i_item_pad=None,
                            i_user_pad=None, i_emb=None, u_emb=None, purpose='readyFORuser').data
        for j in range(valid_test_unique_uids.size()[0]):
            existed_user_id2emb[valid_test_unique_uids[j]] = u_emb[j]

    for i, (valid_test_unique_iids, valid_test_unique_i_item_pad, valid_test_unique_i_user_pad) in tqdm(
            enumerate(readyFORvalidTest_itemloader),
            total=len(readyFORvalidTest_itemloader)):
        i_emb = model.ready(uids=None, iids=valid_test_unique_iids, u_item_pad=None,
                            i_item_pad=valid_test_unique_i_item_pad,
                            i_user_pad=valid_test_unique_i_user_pad, i_emb=None, u_emb=None,
                            purpose='readyFORitem').data
        for j in range(valid_test_unique_iids.size()[0]):
            existed_item_id2emb[valid_test_unique_iids[j]] = i_emb[j]

    with open(out_dir + 'user_and_item_embs.pickle', 'wb') as f:
        pickle.dump([existed_item_id2emb, existed_user_id2emb], f)
    f.close()
    print('existed_item_id2emb.size: ' + str(existed_item_id2emb.size()))
    print('existed_user_id2emb.size: ' + str(existed_user_id2emb.size()))
    result1 = np.array(existed_item_id2emb.cpu())
    np.savetxt(out_dir + 'existed_item_id2emb.txt', result1)
    result2 = np.array(existed_user_id2emb.cpu())
    np.savetxt(out_dir + 'existed_user_id2emb.txt', result2)

    ndcgs, hrs = validate_full(test_loader, model, existed_item_id2emb, existed_user_id2emb,
                               evaluation_interval=5)
    with open(out_dir + 'test_ndcg_hr.txt', 'w') as f:
        for i in range(ndcgs.size()[0]):
            print(str(float(ndcgs[i])) + '\t' + str(float(hrs[i])), file=f)
        f.close()

    print('ndcgs: ', end='\t')
    print(ndcgs)
    print('hrs: ', end='\t')
    print(hrs)
    print('Best epoch: ' + str(ckpt['epoch']) + '\t' + argsStateStr)
    return

if __name__ == '__main__':
    main()
