# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import utils.metric
from utils import *
from models import *
from tqdm import tqdm
import pc_gender_train_lastfm
import pc_age_train_lastfm
import scipy.stats as stats

import pdb
import sys


def train_gcn_baseline(model, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args,LACC):
    optimizer_G = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    [gender,age]=u_sens

    best_perf = 0.0
    for epoch in range(args.num_epochs):
        train_res = {
            'bpr_loss': 0.0,
            'emb_loss': 0.0,
        }
        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)
            j = uij[2].type(torch.long).to(args.device)

            main_user_emb, main_item_emb = model.forward()
            bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            emb_loss = emb_loss * args.l2_reg
            loss = bpr_loss + emb_loss

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['bpr_loss'] += bpr_loss.item()
            train_res['emb_loss'] += emb_loss.item()

        train_res['bpr_loss'] = train_res['bpr_loss'] / len(train_loader)
        train_res['emb_loss'] = train_res['emb_loss'] / len(train_loader)

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        print(training_logs)

        with torch.no_grad():
            t_user_emb, t_item_emb = model.forward()
            test_res = ranking_evaluate(
                user_emb=t_user_emb.detach().cpu().numpy(),
                item_emb=t_item_emb.detach().cpu().numpy(),
                n_users=n_users,
                n_items=n_items,
                train_u2i=train_u2i,
                test_u2i=test_u2i,
                sens=None,
                num_workers=args.num_workers)

        #     precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'gender', 5)
        #     if precision_Att < 0.5:
        #         precision_Att = 1 - precision_Att
        #     precision_Att = round(precision_Att, 4)
        #     test_res['LACC@5-Gen'] = precision_Att
        #     precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'gender', 10)
        #     if precision_Att < 0.5:
        #         precision_Att = 1 - precision_Att
        #     precision_Att = round(precision_Att, 4)
        #     test_res['LACC@10-Gen'] = precision_Att
        #     precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'gender', 20)
        #     if precision_Att < 0.5:
        #         precision_Att = 1 - precision_Att
        #     precision_Att = round(precision_Att, 4)
        #     test_res['LACC@20-Gen'] = precision_Att
        #
        #     precision_age, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'age', 5)
        #     precision_age = round(precision_age, 4)
        #     test_res['LACC@5-age'] = precision_age
        #     precision_age, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'age', 10)
        #     precision_age = round(precision_age, 4)
        #     test_res['LACC@10-age'] = precision_age
        #     precision_age, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'age', 20)
        #     precision_age = round(precision_age, 4)
        #     test_res['LACC@20-age'] = precision_age
        #
        #
        #     ####AUC
        #
        # auc_one, auc_res = pc_gender_train_lastfm.clf_gender_all_pre('auc', epoch, t_user_emb.detach().cpu().numpy(),
        #                                                              gender,
        #                                                              args.emb_size)
        # test_res['AUC'] = round(np.mean(auc_one), 4)
        # f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train_lastfm.clf_age_all_pre('f1', epoch,
        #                                                                   t_user_emb.detach().cpu().numpy(),
        #                                                                          age,
        #                                                                   args.emb_size)
        # f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
        # test_res['Age-F1'] = round(np.mean(f1micro_f1), 4)


        p_eval = ''
        for keys, values in test_res.items():
            p_eval += keys + ':' + '[%.6f]' % values + ' '
        print(p_eval)


        if best_perf < test_res['ndcg@10']:
            best_perf = test_res['ndcg@10']
            torch.save(model, args.param_path)
            print('save successful')


if __name__ == '__main__':


    parser = argparse.ArgumentParser(
        description='lastfm_gcn_baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bakcbone', type=str, default='bpr')
    parser.add_argument('--dataset', type=str, default='./data/lastfm-360k/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--log_path', type=str, default='logs/gcn_base_lastfm.txt')
    parser.add_argument('--param_path', type=str, default='param/bpr_basedir/lastfm/bpr_base_0.04_lastfm.pth')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrain_bprmf_loadpath', type=str, default='param/bpr_basedir/lastfm/bpr_base.pth')
    parser.add_argument('--replace_ratio', type=float, default=0.04)

    args = parser.parse_args()

    #sys.stdout = Logger(args.log_path)
    print(args)

    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_set = pickle.load(f)
        test_u2i = pickle.load(f)
        user_features_dict = pickle.load(f)
        n_users, n_items = pickle.load(f)
    pretrain_bprmf = torch.load(args.pretrain_bprmf_loadpath)
    pretrainuemb, pretrainiemb = pretrain_bprmf.forward()
    pretrainuemb = pretrainuemb.detach().cpu().numpy()
    pretrainiemb = pretrainiemb.detach().cpu().numpy()
    ################非敏感信息侧数据集生成##########
    replace_train_set, replace_train_u2i = non_sevsitive_info_side_dg(pretrainuemb, pretrainiemb, args.replace_ratio,
                                                                      'lastfm')
    print('非敏感信息侧数据集已生成')


    graph = Graph(n_users, n_items, replace_train_u2i, replace_train_u2i)
    norm_adj = graph.generate_ori_norm_adj()

    gender=user_features_dict['gender'].reshape((n_users,1))
    age=user_features_dict['age'].reshape((n_users,1))
    user_feature_n = np.concatenate((gender,age),axis=1)
    user_feature01=np.zeros((user_feature_n.shape[0],5),dtype=np.int32)
    for i in range(user_feature_n.shape[0]):
        user_feature01[i,user_feature_n[i,0]]=1
        user_feature01[i, user_feature_n[i, 1]+2] = 1
    LACC = utils.metric.FairAndPrivacy_lastfm(usernum=n_users, itemnum=n_items, user_feature_n=user_feature_n,
                                       user_feature01=user_feature01, trainset=train_set, train_u2i=train_u2i)

    bprmf = BPRMF(n_users, n_items, args.emb_size, device=args.device)
    #gcn = LightGCN(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)
    
    u_sens = [user_features_dict['gender'].astype(np.int32),user_features_dict['age'].astype(np.int32)]
    dataset = BPRTrainLoader(replace_train_set, replace_train_u2i, train_u2i,n_items)

    train_gcn_baseline(bprmf, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args,LACC)
    #sys.stdout = None
