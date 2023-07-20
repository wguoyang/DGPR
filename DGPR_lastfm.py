import argparse
import time

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


def train_semigcn(gcn, sens, n_users, lr=0.001, num_epochs=1000, device='cpu'):
    sens = torch.tensor(sens).to(torch.long).to(device)
    optimizer = optim.Adam(gcn.parameters(), lr=lr)

    final_loss = 0.0
    for _ in tqdm(range(num_epochs)):
        _, _, su, _ = gcn()
        shuffle_idx = torch.randperm(n_users)
        classify_loss = F.cross_entropy(su[shuffle_idx].squeeze(), sens[shuffle_idx].squeeze())
        optimizer.zero_grad()
        classify_loss.backward()
        optimizer.step()
        final_loss = classify_loss.item()

    print('epoch: %d, classify_loss: %.6f' % (num_epochs, final_loss))


def train_unify_mi(sens_enc, inter_enc, clubu, dataset, u_sens,
                   n_users, n_items, train_u2i, test_u2i, args, LACC, classifier):
    optimizer_G = optim.Adam(inter_enc.parameters(), lr=args.lr)
    club = clubu
    optimizer_D = optim.Adam(club.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    e_su, e_si, presens_u, presens_i = sens_enc.forward()
    e_su = e_su.detach().to(args.device)
    e_si = e_si.detach().to(args.device)
    esii = e_si.detach().cpu().numpy()
    esii = torch.FloatTensor(esii).cuda()
    if args.att == 'gender':
        test_usen = None
        p_su = conditional_samples(e_su.detach().cpu().numpy())
        # p_su = u_sens
        p_si = conditional_samples(e_si.detach().cpu().numpy())
        p_su = torch.tensor(p_su).to(args.device)
        p_si = torch.tensor(p_si).to(args.device)
    elif args.att == 'age' or args.att == 'occ':
        test_usen = None
        p_su = u_sens
        p_su = torch.tensor(p_su).to(args.device)
    # p_si_age = conditional_samples_item_age(presens_i.detach().cpu().numpy())
    # p_si_age = torch.tensor(p_si_age).to(args.device)

    ex_enc = torch.load(args.pretrain_path)
    e_xu, e_xi = ex_enc.forward()
    e_xu = e_xu.detach().to(args.device)
    e_xi = e_xi.detach().to(args.device)
    e_ru_train = get_eru(e_xu, e_xi, e_si, train_u2i, 20, 1000, args.device)
    e_ru = e_ru_train.detach()
    presens_ru = sens_enc.fc(e_ru)
    best_perf = 0.0
    for epoch in range(args.num_epochs):
        train_res = {
            'bpr': 0.0,
            'emb': 0.0,
            'lb': 0.0,
            'ub': 0.0,
            'mi': 0.0,
        }
        if args.att == 'age' or args.att == 'occ':
            p_si = conditional_samples_item_age(presens_i.detach().cpu().numpy())
            p_si = torch.tensor(p_si).to(args.device)
        e_zu, e_zi = inter_enc.forward()
        ############推荐列表层面敏感属性embedding生成与抽样##########
        if args.reclistatt == True:
            # e_ru = get_eru(e_zu, e_zi, e_si, train_u2i, 10, 1000, args.device)
            # e_ru = e_ru.detach()
            # presens_ru = sens_enc.fc(e_ru)
            p_ru = conditional_samples_item_age(presens_ru.detach().cpu().numpy())
            p_ru = torch.tensor(p_ru).to(args.device)
        ##########################################################

        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)
            j = uij[2].type(torch.long).to(args.device)
            main_user_emb, main_item_emb = inter_enc.forward()
            bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            emb_loss = emb_loss * args.l2_reg

            e_zu, e_zi = inter_enc.forward()
            # ############推荐列表层面敏感属性embedding生成与抽样##########
            # if args.reclistatt==True:
            #     e_ru=get_eru(e_zu,e_zi,e_si,train_u2i,10,1000,args.device)
            #     e_ru=e_ru.detach()
            #     presens_ru=sens_enc.fc(e_ru)
            #     p_ru=conditional_samples_item_age(presens_ru.detach().cpu().numpy())
            #     p_ru=torch.tensor(p_ru).to(args.device)
            # ##########################################################
            ########下界###########
            lb1 = condition_info_nce_for_embeddings(e_xu[torch.unique(u)], e_zu[torch.unique(u)],
                                                    e_su[torch.unique(u)], p_su[torch.unique(u)])
            lb2 = condition_info_nce_for_embeddings(e_xi[torch.unique(i)], e_zi[torch.unique(i)],
                                                    e_si[torch.unique(i)], p_si[torch.unique(i)])

            lb = args.lreg * (lb1 + lb2)
            ##########################
            # lb = args.lreg * (lb1 + lb2)
            # our further research found that imposing upper bound constraints on
            # the user-side only gives more stable and better results, so codes has been updated here.
            up = club.forward(e_zu[torch.unique(u)], e_su[torch.unique(u)])

            up = args.ureg * up
            # up = args.ureg * up_gen
            loss = bpr_loss + emb_loss + lb + up

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['bpr'] += bpr_loss.item()
            train_res['emb'] += emb_loss.item()
            train_res['lb'] += lb.item()
            train_res['ub'] += up.item()
        ######gumbel-max-trick拓展到gumbel-topk-trick，利用对抗学习优化e_zu和e_zi###############
        e_zu, e_zi = inter_enc.forward()
        # _,esii,_,_=sens_enc.forward()
        st = time.time()
        e_ru = get_eru_gumbel(e_zu, e_zi, esii, train_u2i, 20)
        en = time.time()
        print(en - st)
        usens = torch.tensor(u_sens).to(torch.long).cuda()
        eru = e_ru.detach()
        opt = classifier.opt
        for i in range(30):  # gender,age30,
            closs = classifier(eru, usens)
            opt.zero_grad()
            closs.backward()
            opt.step()
        classcifyloss = classifier(e_ru, usens)
        classcifyloss = -1 * classcifyloss  # gender,age,occ1
        optimizer_G.zero_grad()
        classcifyloss.backward()
        optimizer_G.step()
        train_res['closs'] = classcifyloss
        ###################################################################################
        train_res['bpr'] = train_res['bpr'] / len(train_loader)
        train_res['emb'] = train_res['emb'] / len(train_loader)
        train_res['lb'] = train_res['lb'] / len(train_loader)
        train_res['ub'] = train_res['ub'] / len(train_loader)

        e_zu, e_zi = inter_enc.forward()

        x_samples = e_zu.detach()
        y_samples = e_su.detach()

        for _ in range(args.train_step):
            mi_loss = club.learning_loss(x_samples, y_samples)
            optimizer_D.zero_grad()
            mi_loss.backward()
            optimizer_D.step()
            train_res['mi'] += mi_loss.item()

        train_res['mi'] = train_res['mi'] / args.train_step

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        print(training_logs)

        with torch.no_grad():

            #################推荐准确性指标################
            t_user_emb, t_item_emb = inter_enc.forward()
            test_res = ranking_evaluate(
                user_emb=t_user_emb.detach().cpu().numpy(),
                item_emb=t_item_emb.detach().cpu().numpy(),
                n_users=n_users,
                n_items=n_items,
                train_u2i=train_u2i,
                test_u2i=test_u2i,
                sens=None,
                num_workers=args.num_workers)
            #################################################

            #################################新增的两个公平评价指标#######################
            ##前一个工作的新评价指标
            if args.att == 'gender':
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
            else :
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

            ####AUC\F1
        if args.att == 'gender':
            auc_one, auc_res = pc_gender_train_lastfm.clf_gender_all_pre('auc', epoch,
                                                                         t_user_emb.detach().cpu().numpy(),
                                                                         usens.detach().cpu().numpy(),
                                                                         args.emb_size)
            test_res['AUC'] = round(np.mean(auc_one), 4)
        else :
            f1p_one, f1r_one, f1res_p, f1res_r = pc_age_train_lastfm.clf_age_all_pre('f1', epoch,
                                                                                     t_user_emb.detach().cpu().numpy(),
                                                                                     usens.detach().cpu().numpy(),
                                                                                     args.emb_size)
            f1micro_f1 = (2 * f1p_one * f1r_one) / (f1p_one + f1r_one)
            test_res['Age-F1'] = round(np.mean(f1micro_f1), 4)

        #############################################################################

        p_eval = ''
        for keys, values in test_res.items():
            p_eval += keys + ':' + '[%.6f]' % values + ' '
        print(p_eval)

        if best_perf < test_res['ndcg@10']:
            best_perf = test_res['ndcg@10']
            torch.save(inter_enc, args.save_path)
            print('save successful')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ml_gcn_fairmi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--dataset', type=str, default='./data/lastfm-360k/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_path', type=str, default='logs/gcn_fairmi.txt')
    parser.add_argument('--param_path', type=str, default='param/gcn_fairmi.pth')
    parser.add_argument('--save_path', type=str, default='param/bpr_basedir/lastfm/DGPR_gcn_age_0.04.pth')
    parser.add_argument('--pretrain_path', type=str, default='param/bpr_basedir/lastfm/gcn_base_0.04_lastfm.pth')
    parser.add_argument('--pretrain_bprmf_loadpath',type=str,default='param/bpr_basedir/lastfm/bpr_base.pth')
    parser.add_argument('--lreg', type=float, default=0.1)
    parser.add_argument('--ureg', type=float,
                        default=0.1)  ##########################################################性别的上界权重要设为1
    parser.add_argument('--train_step', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--att', type=str, default='age')
    parser.add_argument('--replace', type=bool, default=True)
    parser.add_argument('--reclistatt', type=bool, default=False)
    parser.add_argument('--replace_ratio',type=float,default=0.04)

    args = parser.parse_args()

    print(args)
    att = args.att
    ##########原数据集加载#############
    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_set = pickle.load(f)
        test_u2i = pickle.load(f)
        user_features_dict = pickle.load(f)
        n_users, n_items = pickle.load(f)
    ######################################
    #预训练embedding加载
    pretrain_bprmf = torch.load(args.pretrain_bprmf_loadpath)
    pretrainuemb, pretrainiemb = pretrain_bprmf.forward()
    pretrainuemb=pretrainuemb.detach().cpu().numpy()
    pretrainiemb = pretrainiemb.detach().cpu().numpy()
    ################非敏感信息侧数据集生成##########
    replace_train_set, replace_train_u2i=non_sevsitive_info_side_dg(pretrainuemb,pretrainiemb,args.replace_ratio,'lastfm')
    print('非敏感信息侧数据集已生成')
    ######################################
    if args.replace == True:
        dataset = BPRTrainLoader(replace_train_set, replace_train_u2i, train_u2i, n_items)
        graph = Graph(n_users, n_items, replace_train_u2i, replace_train_u2i)
        norm_adj = graph.generate_ori_norm_adj()
    else:
        dataset = BPRTrainLoader(train_set, train_u2i, train_u2i, n_items)
        graph = Graph(n_users, n_items, train_u2i, train_u2i)
        norm_adj = graph.generate_ori_norm_adj()



    # user_side_features = np.load('./data/ml-1m/users_features_3num.npy', allow_pickle=True)
    # if att == 'gender':
    #     u_sens = user_side_features[:, 0]
    # elif att == 'age':
    #     u_sens = user_side_features[:, 1]
    # elif att=='occ':
    #     u_sens = user_side_features[:, 2]
    # else:
    #     u_sens=user_side_features
    if att == 'gender':
        u_sens=user_features_dict['gender']
    else:
        u_sens = user_features_dict['age']
    #########生成新的用于敏感属性编码器数据集和构建敏感属性编码器############
    if att=='gender':
        new_train_u2i = sevsitive_gender_side_dg(pretrainuemb,pretrainiemb,args.replace_ratio,u_sens,att,'lastfm')
    else:
        new_train_u2i = sevsitive_ageocc_side_dg(pretrainuemb, pretrainiemb, args.replace_ratio, u_sens, att, 'lastfm')
    print('敏感信息侧数据集已生成')

    if args.replace == True:
        graph_sens = Graph(n_users, n_items, new_train_u2i, new_train_u2i)
    else:
        graph_sens = Graph(n_users, n_items, train_u2i, train_u2i)
    norm_adj_sens = graph_sens.generate_ori_norm_adj()
    sens_enc = SemiGCN(n_users, n_items, norm_adj_sens,
                       args.emb_size, args.n_layers, args.device,
                       nb_classes=np.unique(u_sens).shape[0])  # nb_classes=7
    #########################################################################
    ###############构建LACC指标##############
    gender = user_features_dict['gender'].reshape((n_users, 1))
    age = user_features_dict['age'].reshape((n_users, 1))
    user_feature_n = np.concatenate((gender, age), axis=1)
    user_feature01 = np.zeros((user_feature_n.shape[0], 5), dtype=np.int32)
    for i in range(user_feature_n.shape[0]):
        user_feature01[i, user_feature_n[i, 0]] = 1
        user_feature01[i, user_feature_n[i, 1] + 2] = 1
    LACC = utils.metric.FairAndPrivacy_lastfm(usernum=n_users, itemnum=n_items, user_feature_n=user_feature_n,
                                              user_feature01=user_feature01, trainset=train_set, train_u2i=train_u2i)
    ########################################
    if args.model=='gcn':
        inter_enc = LightGCN(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)
    else:
        inter_enc = BPRMF(n_users, n_items, args.emb_size, args.device)
    club = CLUBSample(args.emb_size, args.emb_size, args.hidden_size, args.device)
    # classifier=classifiergender().cuda()
    if att=='gender':
        classifier = classifiergender().cuda()
    else:
        classifier = classifierage_occ(outdim=3).cuda()
    train_semigcn(sens_enc, u_sens, n_users, device=args.device)
    train_unify_mi(sens_enc, inter_enc, club, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args, LACC,
                   classifier)