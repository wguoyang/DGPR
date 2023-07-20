import argparse
import time
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
    [club1,club2,club3,conpu,conpi] = clubu
    [sens_enc1,sens_enc2,sens_enc3]=sens_enc
    [classifier1,classifier2,classifier3]=classifier
    optimizer_D1 = optim.Adam(club1.parameters(), lr=args.lr)
    optimizer_D2 = optim.Adam(club2.parameters(), lr=args.lr)
    optimizer_D3 = optim.Adam(club3.parameters(), lr=args.lr)
    optimizer_conpu = optim.Adam(conpu.parameters(), lr=args.lr)
    optimizer_conpi = optim.Adam(conpi.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    ######gender###
    e_su_gen, e_si_gen, presens_u_gen, presens_i_gen = sens_enc1.forward()
    e_su_gen = e_su_gen.detach().to(args.device)
    e_si_gen = e_si_gen.detach().to(args.device)
    esii_gen = e_si_gen.detach().cpu().numpy()
    esii_gen = torch.FloatTensor(esii_gen).cuda()#####用于生成推荐列表敏感属性表示的物品敏感属性向量
    p_su_gen = conditional_samples(e_su_gen.detach().cpu().numpy())
    p_si_gen = conditional_samples(e_si_gen.detach().cpu().numpy())
    p_su_gen = torch.tensor(p_su_gen).to(args.device)
    p_si_gen = torch.tensor(p_si_gen).to(args.device)
    #######age#####
    e_su_age, e_si_age, presens_u_age, presens_i_age = sens_enc2.forward()
    e_su_age = e_su_age.detach().to(args.device)
    e_si_age = e_si_age.detach().to(args.device)
    esii_age = e_si_age.detach().cpu().numpy()
    esii_age = torch.FloatTensor(esii_age).cuda()  #####用于生成推荐列表敏感属性表示的物品敏感属性向量
    p_su_age = u_sens[:,1]
    p_su_age = torch.tensor(p_su_age).to(args.device)
    #########occ#########
    e_su_occ, e_si_occ, presens_u_occ, presens_i_occ = sens_enc3.forward()
    e_su_occ = e_su_occ.detach().to(args.device)
    e_si_occ = e_si_occ.detach().to(args.device)
    esii_occ = e_si_occ.detach().cpu().numpy()
    esii_occ = torch.FloatTensor(esii_occ).cuda()  #####用于生成推荐列表敏感属性表示的物品敏感属性向量
    p_su_occ = u_sens[:, 2]
    p_su_occ = torch.tensor(p_su_occ).to(args.device)

    # e_su, e_si, presens_u, presens_i = sens_enc.forward()
    # e_su = e_su.detach().to(args.device)
    # e_si = e_si.detach().to(args.device)
    # esii = e_si.detach().cpu().numpy()
    # esii = torch.FloatTensor(esii).cuda()
    # if args.att == 'gender':
    #     test_usen = None
    #     p_su = conditional_samples(e_su.detach().cpu().numpy())
    #     # p_su = u_sens
    #     p_si = conditional_samples(e_si.detach().cpu().numpy())
    #     p_su = torch.tensor(p_su).to(args.device)
    #     p_si = torch.tensor(p_si).to(args.device)
    # elif args.att == 'age' or args.att == 'occ':
    #     test_usen = None
    #     p_su = u_sens
    #     p_su = torch.tensor(p_su).to(args.device)
    # p_si_age = conditional_samples_item_age(presens_i.detach().cpu().numpy())
    # p_si_age = torch.tensor(p_si_age).to(args.device)
    #
    ex_enc = torch.load(args.pretrain_path)
    e_xu, e_xi = ex_enc.forward()
    e_xu = e_xu.detach().to(args.device)
    e_xi = e_xi.detach().to(args.device)
    # e_ru_train = get_eru(e_xu, e_xi, e_si, train_u2i, 20, 1000, args.device)
    # e_ru = e_ru_train.detach()
    # presens_ru = sens_enc.fc(e_ru)
    best_perf = 0.0
    for epoch in range(args.num_epochs):
        train_res = {
            'bpr': 0.0,
            'emb': 0.0,
            'lb': 0.0,
            'ub': 0.0,
            'mi': 0.0,
        }

        p_si_age = conditional_samples_item_age(presens_i_age.detach().cpu().numpy())
        p_si_age = torch.tensor(p_si_age).to(args.device)
        p_si_occ = conditional_samples_item_age(presens_i_occ.detach().cpu().numpy())
        p_si_occ = torch.tensor(p_si_occ).to(args.device)


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
            # lb1 = condition_info_nce_for_embeddings(e_xu[torch.unique(u)], e_zu[torch.unique(u)],
            #                                         e_su_gen[torch.unique(u)], p_su_gen[torch.unique(u)])
            # lb2 = condition_info_nce_for_embeddings(e_xi[torch.unique(i)], e_zi[torch.unique(i)],
            #                                         e_si_gen[torch.unique(i)], p_si_gen[torch.unique(i)])
            # lb3 = condition_info_nce_for_embeddings(e_xu[torch.unique(u)], e_zu[torch.unique(u)],
            #                                         e_su_age[torch.unique(u)], p_su_age[torch.unique(u)])
            # lb4 = condition_info_nce_for_embeddings(e_xi[torch.unique(i)], e_zi[torch.unique(i)],
            #                                         e_si_age[torch.unique(i)], p_si_age[torch.unique(i)])
            # lb5 = condition_info_nce_for_embeddings(e_xu[torch.unique(u)], e_zu[torch.unique(u)],
            #                                         e_su_occ[torch.unique(u)], p_su_occ[torch.unique(u)])
            # lb6 = condition_info_nce_for_embeddings(e_xi[torch.unique(i)], e_zi[torch.unique(i)],
            #                                         e_si_occ[torch.unique(i)], p_si_occ[torch.unique(i)])
            w=[2,1,1]####控制不同敏感属性之间的权重
            sum=np.sum(w)
            e_su_com=(e_su_gen+e_su_age+e_su_occ)/3
            e_si_com=(e_si_gen+e_si_age+e_si_occ)/3
            pu_sample=conpu.forward(e_su_com)
            pi_sample=conpi.forward(e_si_com)
            lb1=lower_conditionp(e_xu[torch.unique(u)],pu_sample[torch.unique(u)],e_zu[torch.unique(u)],e_su_com[torch.unique(u)])
            lb2=lower_conditionp(e_xi[torch.unique(i)],pi_sample[torch.unique(i)],e_zi[torch.unique(i)],e_si_com[torch.unique(i)])
            #lb = args.lreg * (w[0]*(lb1 + lb2)+w[1]*(lb3 + lb4)+w[2]*(lb5 + lb6))/sum
            ##########################
            lb = args.lreg * (lb1 + lb2)
            # our further research found that imposing upper bound constraints on
            # the user-side only gives more stable and better results, so codes has been updated here.
            up1 = club1.forward(e_zu[torch.unique(u)], e_su_gen[torch.unique(u)])
            up2 = club2.forward(e_zu[torch.unique(u)], e_su_age[torch.unique(u)])
            up3 = club3.forward(e_zu[torch.unique(u)], e_su_occ[torch.unique(u)])

            up = args.ureg * (w[0]*up1+w[1]*up2+w[2]*up3)/sum
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
        esii=torch.cat([esii_gen,esii_age,esii_occ],dim=1)
        e_ru = get_eru_gumbel(e_zu, e_zi, esii, train_u2i, 20)
        e_ru_gen,e_ru_age,e_ru_occ=torch.split(e_ru,[64,64,64],dim=1)
        en = time.time()
        print(en - st)
        usens_gen = u_sens[:,0]
        usens_age = u_sens[:, 1]
        usens_occ = u_sens[:, 2]
        usens_gen = torch.tensor(usens_gen).to(torch.long).cuda()
        usens_age = torch.tensor(usens_age).to(torch.long).cuda()
        usens_occ = torch.tensor(usens_occ).to(torch.long).cuda()
        eru_gen = e_ru_gen.detach()
        eru_age = e_ru_age.detach()
        eru_occ = e_ru_occ.detach()
        opt1 = classifier1.opt
        opt2 = classifier2.opt
        opt3 = classifier3.opt
        ########训练k个分类器#####
        for i in range(30):  # gender,age30,
            closs1 = classifier1(eru_gen, usens_gen)
            opt1.zero_grad()
            closs1.backward()
            opt1.step()
            closs2 = classifier2(eru_age, usens_age)
            opt2.zero_grad()
            closs2.backward()
            opt2.step()
            closs3 = classifier3(eru_occ, usens_occ)
            opt3.zero_grad()
            closs3.backward()
            opt3.step()
        #######################
        ###########分类器前向传播得到损失#######
        classcifyloss1 = classifier1(e_ru_gen, usens_gen)
        classcifyloss2 = classifier2(e_ru_age, usens_age)
        classcifyloss3 = classifier3(e_ru_occ, usens_occ)
        classcifyloss = -1 * (w[0]*classcifyloss1+w[1]*classcifyloss2+w[2]*classcifyloss3)  # gender,age,occ1
        ##############分类器损失的负值更新embedding#########
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
        y_samples1 = e_su_gen.detach()
        y_samples2 = e_su_age.detach()
        y_samples3 = e_su_occ.detach()
        e_su_com=(y_samples1+y_samples2+y_samples3)/3
        x_samples1 = e_si_gen.detach()
        x_samples2 = e_si_age.detach()
        x_samples3 = e_si_occ.detach()
        e_si_com=(x_samples1+x_samples2+x_samples3)/3
        conyu=e_xu.detach()
        conyi = e_xi.detach()

        for _ in range(args.train_step):
            mi_loss1 = club1.learning_loss(x_samples, y_samples1)
            optimizer_D1.zero_grad()
            mi_loss1.backward()
            optimizer_D1.step()
            mi_loss2 = club2.learning_loss(x_samples, y_samples2)
            optimizer_D2.zero_grad()
            mi_loss2.backward()
            optimizer_D2.step()
            mi_loss3 = club3.learning_loss(x_samples, y_samples3)
            optimizer_D3.zero_grad()
            mi_loss3.backward()
            optimizer_D3.step()
            mi_loss=mi_loss1+mi_loss2+mi_loss3
            train_res['mi'] += mi_loss.item()
            conploss1=conpu.learning_loss(e_su_com,conyu)
            optimizer_conpu.zero_grad()
            conploss1.backward()
            optimizer_conpu.step()
            conploss2 = conpi.learning_loss(e_si_com, conyi)
            optimizer_conpi.zero_grad()
            conploss2.backward()
            optimizer_conpi.step()


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

        ####AUC\F1

        auc_one, auc_res = pc_gender_train.clf_gender_all_pre('auc', epoch, t_user_emb.detach().cpu().numpy(),
                                                              args.emb_size, args.device)
        test_res['Gen-AUC'] = round(np.mean(auc_one), 4)

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
        #############################################################################

        p_eval = ''
        for keys, values in test_res.items():
            p_eval += keys + ':' + '[%.6f]' % values + ' '
        print(p_eval)

        # if best_perf < test_res['ndcg@10']:
        #     best_perf = test_res['ndcg@10']
        #     torch.save(inter_enc, args.save_path)
        #     print('save successful')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ml_gcn_fairmi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, default='bprmf')
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_path', type=str, default='logs/gcn_fairmi.txt')
    parser.add_argument('--param_path', type=str, default='param/gcn_fairmi.pth')
    parser.add_argument('--save_path', type=str, default='param/bpr_basedir/ml1m/DGPR_gcn_com_0.02.pth')
    parser.add_argument('--pretrain_path', type=str, default='param/bpr_basedir/ml1m/new_bpr_base_0.02_ml1m.pth')
    parser.add_argument('--pretrain_bprmf_loadpath',type=str,default='param/bpr_base.pth')
    parser.add_argument('--lreg', type=float, default=0.2)
    parser.add_argument('--ureg', type=float,default=10)  ##########################################################性别的上界权重要设为1
    parser.add_argument('--train_step', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--att', type=str, default='com')
    parser.add_argument('--replace', type=bool, default=True)
    parser.add_argument('--reclistatt', type=bool, default=False)
    parser.add_argument('--replace_ratio',type=float,default=0.02)

    args = parser.parse_args()

    print(args)
    att = args.att
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
    ######################################
    #预训练embedding加载
    pretrain_bprmf = torch.load(args.pretrain_bprmf_loadpath)
    pretrainuemb, pretrainiemb = pretrain_bprmf.forward()
    pretrainuemb=pretrainuemb.detach().cpu().numpy()
    pretrainiemb = pretrainiemb.detach().cpu().numpy()
    ################非敏感信息侧数据集生成##########
    replace_train_set, replace_train_u2i=non_sevsitive_info_side_dg(pretrainuemb,pretrainiemb,args.replace_ratio,'ml1m')
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



    user_side_features = np.load('./data/ml-1m/users_features_3num.npy', allow_pickle=True)
    u_sens=user_side_features



    #########生成新的用于敏感属性编码器数据集和构建敏感属性编码器############
    new_train_u2i_gen = sevsitive_gender_side_dg(pretrainuemb,pretrainiemb,args.replace_ratio,u_sens[:, 0],'gender','ml1m')
    new_train_u2i_age = sevsitive_ageocc_side_dg(pretrainuemb, pretrainiemb, args.replace_ratio, u_sens[:, 1], 'age', 'ml1m')
    new_train_u2i_occ = sevsitive_ageocc_side_dg(pretrainuemb, pretrainiemb, args.replace_ratio, u_sens[:, 2], 'occ', 'ml1m')
    print('敏感信息侧数据集已生成')


    graph_sens1 = Graph(n_users, n_items, new_train_u2i_gen, new_train_u2i_gen)
    graph_sens2 = Graph(n_users, n_items, new_train_u2i_age, new_train_u2i_age)
    graph_sens3 = Graph(n_users, n_items, new_train_u2i_occ, new_train_u2i_occ)
    norm_adj_sens1 = graph_sens1.generate_ori_norm_adj()
    sens_enc1 = SemiGCN(n_users, n_items, norm_adj_sens1,
                       args.emb_size, args.n_layers, args.device,
                       nb_classes=np.unique(u_sens[:,0]).shape[0])  # nb_classes=7
    norm_adj_sens2 = graph_sens2.generate_ori_norm_adj()
    sens_enc2 = SemiGCN(n_users, n_items, norm_adj_sens2,
                        args.emb_size, args.n_layers, args.device,
                        nb_classes=np.unique(u_sens[:, 1]).shape[0])
    norm_adj_sens3 = graph_sens3.generate_ori_norm_adj()
    sens_enc3 = SemiGCN(n_users, n_items, norm_adj_sens3,
                        args.emb_size, args.n_layers, args.device,
                        nb_classes=np.unique(u_sens[:, 2]).shape[0])
    #########################################################################
    ###############构建LACC指标##############
    user_feature_n = np.load('./data/ml-1m/users_features_3num.npy', allow_pickle=True)
    user_feature01 = np.load('./data/ml-1m/users_features_list.npy', allow_pickle=True)
    LACC = utils.metric.FairAndPrivacy(usernum=n_users, itemnum=n_items, user_feature_n=user_feature_n,
                                       user_feature01=user_feature01, trainset=train_set, train_u2i=train_u2i)
    ########################################
    if args.model=='gcn':
        inter_enc = LightGCN(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)
        #inter_enc = torch.load('param/bpr_basedir/ml1m/DGPR_gcn_com_0.02.pth')
    else:
        inter_enc = BPRMF(n_users, n_items, args.emb_size, args.device)
    club1 = CLUBSample(args.emb_size, args.emb_size, args.hidden_size, args.device)
    club2 = CLUBSample(args.emb_size, args.emb_size, args.hidden_size, args.device)
    club3 = CLUBSample(args.emb_size, args.emb_size, args.hidden_size, args.device)
    # classifier=classifiergender().cuda()
    train_semigcn(sens_enc1, u_sens[:,0], n_users, device=args.device)
    train_semigcn(sens_enc2, u_sens[:,1], n_users, device=args.device)
    train_semigcn(sens_enc3, u_sens[:,2], n_users, device=args.device)
    classifier1 = classifiergender().cuda()
    classifier2 = classifierage_occ(outdim=7).cuda()
    classifier3 = classifierage_occ(outdim=21).cuda()
    sens_enc=[sens_enc1,sens_enc2,sens_enc3]
    classifier=[classifier1,classifier2,classifier3]
    conditionpu=conditionp_app(args.emb_size, args.emb_size, args.hidden_size, args.device)
    conditionpi = conditionp_app(args.emb_size, args.emb_size, args.hidden_size, args.device)
    club=[club1,club2,club3,conditionpu,conditionpi]
    #train_semigcn(sens_enc, u_sens, n_users, device=args.device)
    train_unify_mi(sens_enc, inter_enc, club, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args, LACC,
                   classifier)