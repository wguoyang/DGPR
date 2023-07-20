# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='1'
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
import pc_gender_train
import pc_age_train
import pc_occ_train
import scipy.stats as stats

import pdb
import sys


def train_gcn_baseline(model, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args,LACC):
    optimizer_G = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    #add_train_loader = DataLoader(addtrainset, shuffle=True, batch_size=256, num_workers=args.num_workers)

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

        # for uij in add_train_loader:
        #     u = uij[0].type(torch.long).to(args.device)
        #     i = uij[1].type(torch.long).to(args.device)
        #     j = uij[2].type(torch.long).to(args.device)
        #     all_user_emb=model.embeddings['user_embeddings'].weight
        #     all_item_emb=model.embeddings['item_embeddings'].weight
        #     all_noise_emb=model.noise_item.weight
        #     noise_emb=all_noise_emb+all_item_emb
        #
        #     main_user_emb, main_item_emb = model.propagate(model.norm_adj,all_user_emb,noise_emb)
        #     bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
        #     emb_loss = emb_loss * args.l2_reg + 10000*(all_noise_emb[i].norm(2).pow(2)/i.shape[0])
        #     loss = bpr_loss + emb_loss
        #
        #     optimizer_G.zero_grad()
        #     loss.backward()
        #     optimizer_G.step()
        #
        #     train_res['bpr_loss'] += bpr_loss.item()
        #     train_res['emb_loss'] += emb_loss.item()

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

            ##前一个工作的新评价指标
        #
        #     precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'gender',5)
        #     if precision_Att < 0.5:
        #         precision_Att = 1 - precision_Att
        #     precision_Att = round(precision_Att, 4)
        #     test_res['LACC@5-Gen'] = precision_Att
        #     precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'gender',10)
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
        #                                               t_item_emb.detach().cpu().numpy(), 'age',5)
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
        #     precision_occ, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'occ',5)
        #     precision_occ = round(precision_occ, 4)
        #     test_res['LACC@5-occ'] = precision_occ
        #     precision_occ, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'occ',10)
        #     precision_occ = round(precision_occ, 4)
        #     test_res['LACC@10-occ'] = precision_occ
        #     precision_occ, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
        #                                               t_item_emb.detach().cpu().numpy(), 'occ', 20)
        #     precision_occ = round(precision_occ, 4)
        #     test_res['LACC@20-occ'] = precision_occ
        #
        #     ####AUC
        # auc_one, auc_res = pc_gender_train.clf_gender_all_pre('auc', epoch, t_user_emb.detach().cpu().numpy(),
        #                                                       args.emb_size, args.device)
        # # auc_one, auc_res = pc_gender_train_lastfm.clf_gender_all_pre('auc', epoch, t_user_emb.detach().cpu().numpy(),
        # #                                                              u_sens,
        # #                                                              args.emb_size)
        # test_res['AUC'] = round(np.mean(auc_one), 4)
        # f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train.clf_age_all_pre('f1', epoch,
        #                                                                   t_user_emb.detach().cpu().numpy(),
        #                                                                   args.emb_size, args.device)
        # f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
        # test_res['Age-F1'] = round(np.mean(f1micro_f1), 4)
        # f1p_one, f1r_one, f1res_p, f1res_r = pc_occ_train.clf_occ_all_pre('f1', epoch,
        #                                                                   t_user_emb.detach().cpu().numpy(),
        #                                                                   args.emb_size, args.device)
        # f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
        # test_res['Occ-F1'] = round(np.mean(f1micro_f1), 4)


        p_eval = ''
        for keys, values in test_res.items():
            p_eval += keys + ':' + '[%.6f]' % values + ' '
        print(p_eval)

        if best_perf < test_res['ndcg@10']:
            best_perf = test_res['ndcg@10']
            torch.save(model, args.save_path)
            print('save successful')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ml_gcn_baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bakcbone', type=str, default='gcn')
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)#lastfm4096,ml1m2048
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--log_path', type=str, default='logs/gcn_base.txt')
    parser.add_argument('--param_path', type=str, default='param/gcn_base.pth')
    parser.add_argument('--save_path', type=str, default='param/bpr_basedir/ml1m/new_gcn_base_0.10_ml1m.pth')
    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--att', type=str, default='gender')
    parser.add_argument('--pretrain_bprmf_loadpath', type=str, default='param/bpr_base.pth')
    parser.add_argument('--replace_ratio', type=float, default=0.10)

    args = parser.parse_args()

    #sys.stdout = Logger(args.log_path)
    print(args)
    ##########原数据集加载#############
    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)
    ##########################################

    pretrain_bprmf = torch.load(args.pretrain_bprmf_loadpath)
    pretrainuemb, pretrainiemb = pretrain_bprmf.forward()
    pretrainuemb = pretrainuemb.detach().cpu().numpy()
    pretrainiemb = pretrainiemb.detach().cpu().numpy()
    ################非敏感信息侧数据集生成##########
    replace_train_set, replace_train_u2i = non_sevsitive_info_side_dg(pretrainuemb, pretrainiemb, args.replace_ratio,
                                                                      'ml1m')
    print('非敏感信息侧数据集已生成')

    ################修正后的数据集#############
    # with open("./DGmodel/output/bns_replace_ratio0.02.pkl", 'rb') as f:
    #     new_ui_trainset=pickle.load(f)
    # new_train_set=defaultdict(list)
    # for user in range(n_users):
    #     items=new_ui_trainset[user].tolist()
    #     new_train_set[user]=items
    # replace_train_u2i=new_train_set############
    # train_dict={}
    # trainuser=[]
    # trainitem=[]
    # trainrating=[]
    # for user in range(n_users):
    #     for item in new_ui_trainset[user]:
    #         trainuser.append(user)
    #         trainitem.append(item)
    #         trainrating.append(1)
    # trainrating=np.array(trainrating)
    # trainitem=np.array(trainitem)
    # trainuser=np.array(trainuser)
    # train_dict['userid']=trainuser
    # train_dict['itemid']=trainitem
    # train_dict['rating']=trainrating
    # replace_train_set=train_dict##############
    ##########################################

    ###############构建LACC指标ml1m##############
    user_feature_n = np.load('./data/ml-1m/users_features_3num.npy', allow_pickle=True)
    user_feature01 = np.load('./data/ml-1m/users_features_list.npy', allow_pickle=True)
    LACC = utils.metric.FairAndPrivacy( usernum=n_users, itemnum=n_items, user_feature_n=user_feature_n,
                                       user_feature01=user_feature01, trainset=train_set,train_u2i=train_u2i)
    ########################################

    ###############构建LACC指标lastfm##############
    # user_feature_n = user_side_features['gender'].astype(np.int32)
    # user_feature01=np.zeros((user_feature_n.shape[0],2),dtype=np.int32)
    # for i in range(user_feature_n.shape[0]):
    #     user_feature01[i,user_feature_n[i]]=1
    # user_feature_n=user_feature_n.reshape(-1,1)
    # LACC = utils.metric.FairAndPrivacy(usernum=n_users, itemnum=n_items, user_feature_n=user_feature_n,
    #                                    user_feature01=user_feature01, trainset=train_set, train_u2i=train_u2i)
    ########################################


    # bprmf = BPRMF(n_users, n_items, args.emb_size, device=args.device)
    graph = Graph(n_users, n_items, replace_train_u2i,replace_train_u2i)
    norm_adj = graph.generate_ori_norm_adj()

    gcn = LightGCN(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)
    #gcn=torch.load('param/bpr_basedir/lastfm/new_gcn_base_0.04_lastfm.pth')
    # t_user_emb, t_item_emb = gcn.forward()
    # test_res = {}
    # if args.att == 'gender':
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
    #     auc_one, auc_res = pc_gender_train.clf_gender_all_pre('auc', 0, t_user_emb.detach().cpu().numpy(),
    #                                                           args.emb_size, args.device)
    #     print(auc_one)
    #
    # elif args.att == 'age':
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
    #     f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train.clf_age_all_pre('f1', 0,
    #                                                                       t_user_emb.detach().cpu().numpy(),
    #                                                                       args.emb_size, args.device)
    #     f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
    #     print(f1micro_f1)
    # else:
    #     precision_occ, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
    #                                               t_item_emb.detach().cpu().numpy(), 'occ', 5)
    #     precision_occ = round(precision_occ, 4)
    #     test_res['LACC@5-occ'] = precision_occ
    #     precision_occ, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
    #                                               t_item_emb.detach().cpu().numpy(), 'occ', 10)
    #     precision_occ = round(precision_occ, 4)
    #     test_res['LACC@10-occ'] = precision_occ
    #     precision_occ, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
    #                                               t_item_emb.detach().cpu().numpy(), 'occ', 20)
    #     precision_occ = round(precision_occ, 4)
    #     test_res['LACC@20-occ'] = precision_occ
    #     f1p_one, f1r_one, f1res_p, f1res_r = pc_occ_train.clf_occ_all_pre('f1', 0,
    #                                                                       t_user_emb.detach().cpu().numpy(),
    #                                                                       args.emb_size, args.device)
    #     f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
    #     print(f1micro_f1)
    # print(test_res)
    u_sens = user_side_features['gender'].astype(np.int32)
    dataset = BPRTrainLoader(replace_train_set, replace_train_u2i, train_u2i,n_items)
    #addset=BPRTrainLoader_add(replace_train_set,train_u2i,replace_train_u2i,n_items)

    train_gcn_baseline(gcn, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args,LACC)
    #sys.stdout = None
