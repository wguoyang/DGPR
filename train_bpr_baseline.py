# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""
import argparse

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
from utils import *
from models import *
from tqdm import tqdm
import pc_gender_train
import pc_age_train
import pc_occ_train
import scipy.stats as stats

import pdb
import sys


def train_bprmf_baseline(model, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args,LACC):
    optimizer_G = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

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
        u_sens=None
        with torch.no_grad():
            t_user_emb, t_item_emb = model.forward()
            test_res = ranking_evaluate(
                user_emb=t_user_emb.detach().cpu().numpy(),
                item_emb=t_item_emb.detach().cpu().numpy(),
                n_users=n_users,
                n_items=n_items,
                train_u2i=train_u2i,
                test_u2i=test_u2i,
                sens=u_sens,
                num_workers=args.num_workers)
            precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'gender', 5)
            if precision_Att < 0.5:
                precision_Att = 1 - precision_Att
            precision_Att = round(precision_Att, 4)
            test_res['LACC@5-Gen'] = precision_Att
            precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'gender', 10)
            if precision_Att < 0.5:
                precision_Att = 1 - precision_Att
            precision_Att = round(precision_Att, 4)
            test_res['LACC@10-Gen'] = precision_Att
            precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'gender', 20)
            if precision_Att < 0.5:
                precision_Att = 1 - precision_Att
            precision_Att = round(precision_Att, 4)
            test_res['LACC@20-Gen'] = precision_Att

            precision_age, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'age', 5)
            precision_age = round(precision_age, 4)
            test_res['LACC@5-age'] = precision_age
            precision_age, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'age', 10)
            precision_age = round(precision_age, 4)
            test_res['LACC@10-age'] = precision_age
            precision_age, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'age', 20)
            precision_age = round(precision_age, 4)
            test_res['LACC@20-age'] = precision_age

            precision_occ, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'occ', 5)
            precision_occ = round(precision_occ, 4)
            test_res['LACC@5-occ'] = precision_occ
            precision_occ, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'occ', 10)
            precision_occ = round(precision_occ, 4)
            test_res['LACC@10-occ'] = precision_occ
            precision_occ, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'occ', 20)
            precision_occ = round(precision_occ, 4)
            test_res['LACC@20-occ'] = precision_occ

            ####AUC
        auc_one, auc_res = pc_gender_train.clf_gender_all_pre('auc', epoch, t_user_emb.detach().cpu().numpy(),
                                                              args.emb_size, args.device)
        test_res['AUC'] = round(np.mean(auc_one), 4)
        f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train.clf_age_all_pre('f1', epoch,
                                                                          t_user_emb.detach().cpu().numpy(),
                                                                          args.emb_size, args.device)
        f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
        test_res['Age-F1'] = round(np.mean(f1micro_f1), 4)
        f1p_one, f1r_one, f1res_p, f1res_r = pc_occ_train.clf_occ_all_pre('f1', epoch,
                                                                          t_user_emb.detach().cpu().numpy(),
                                                                          args.emb_size, args.device)
        f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
        test_res['Occ-F1'] = round(np.mean(f1micro_f1), 4)

        p_eval = ''
        for keys, values in test_res.items():
            p_eval += keys + ':' + '[%.6f]' % values + ' '
        print(p_eval)

        # if best_perf < test_res['ndcg@10']:
        #     best_perf = test_res['ndcg@10']
        #     torch.save(model, args.param_path)
        #     print('save successful')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ml_bpr_baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bakcbone', type=str, default='bpr')
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--log_path', type=str, default='logs/bpr_base.txt')
    parser.add_argument('--param_path', type=str, default='param/new_bpr_base_0.02_ml1m.pth')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    #sys.stdout = Logger(args.log_path)
    print(args)

    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)

    ################修正后的数据集#############
    with open("./DGmodel/output/ml-1m-replace0.2-privacy0.5.pkl", 'rb') as f:
        new_ui_trainset = pickle.load(f)
    new_train_set = defaultdict(list)
    for user in range(n_users):
        items = new_ui_trainset[user].tolist()
        new_train_set[user] = items
    train_u2i = new_train_set  ############
    train_dict = {}
    trainuser = []
    trainitem = []
    trainrating = []
    for user in range(n_users):
        for item in new_ui_trainset[user]:
            trainuser.append(user)
            trainitem.append(item)
            trainrating.append(1)
    trainrating = np.array(trainrating)
    trainitem = np.array(trainitem)
    trainuser = np.array(trainuser)
    train_dict['userid'] = trainuser
    train_dict['itemid'] = trainitem
    train_dict['rating'] = trainrating
    train_set = train_dict  ##############
    ##########################################

    bprmf = BPRMF(n_users, n_items, args.emb_size, device=args.device)
    u_sens = user_side_features['gender'].astype(np.int32)
    dataset = BPRTrainLoader(train_set, train_u2i, train_u2i,n_items)
    ###############构建LACC指标##############
    user_feature_n = np.load('./data/ml-1m/users_features_3num.npy', allow_pickle=True)
    user_feature01 = np.load('./data/ml-1m/users_features_list.npy', allow_pickle=True)
    LACC = utils.metric.FairAndPrivacy(usernum=n_users, itemnum=n_items, user_feature_n=user_feature_n,
                                       user_feature01=user_feature01, trainset=train_set, train_u2i=train_u2i)
    ########################################

    train_bprmf_baseline(bprmf, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args,LACC)
    #sys.stdout = None
