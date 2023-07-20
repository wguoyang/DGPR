# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

import os
#os.environ["CUDA_VISIBLE_DEVICES"] ='1'


from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch.nn.functional as F
import torch.autograd as autograd 
import torch.utils.data as data

from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score
import pdb
import copy 
from collections import defaultdict
import time





class Classifier(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(Classifier, self).__init__()
        """ 
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        three module: LineGCN, AvgReadout, Discriminator
        """     
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        # self.net1 = nn.Sequential(
        #         nn.Linear(self.embed_dim, int(self.embed_dim*2), bias=True),
        #         nn.LeakyReLU(0.2),
        #         # nn.Sigmoid(),
        #         nn.Dropout(p=0.2),
        #         nn.Linear(int(self.embed_dim*2), self.embed_dim, bias=True),
        #         nn.LeakyReLU(0.2),
        #         nn.Dropout(p=0.2),
        #         nn.Linear(self.embed_dim, int(self.embed_dim/2), bias=True),
        #         nn.LeakyReLU(0.2),
        #         nn.Dropout(p=0.2),
        #         nn.Linear(int(self.embed_dim/2), self.out_dim, bias=True),
        #         nn.LeakyReLU(0.2),
        #         nn.Dropout(p=0.2),
        #         nn.Linear(self.out_dim, self.out_dim, bias=True)
        #     )
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim/2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim/2), int(self.embed_dim/4), bias=True), 
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout(p=0.3),
            # nn.Sigmoid(),
            nn.Linear(int(self.embed_dim /4), self.out_dim , bias=True),
            # nn.Sigmoid(),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(self.out_dim, self.out_dim , bias=True),
            #nn.Sigmoid()
            )
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()
    def forward(self, emb0,label0):
        scores = self.net(emb0)
        outputs = F.log_softmax(scores, dim=1)
        label0 =label0.view(-1)#展开成一行
        loss = self.criterion(outputs, label0.long())
        # pdb.set_trace()
        return loss 

    def prediction(self, emb0): 
        scores = self.net(emb0)
        outputs = F.log_softmax(scores, dim=1)
        #（1）dim=0：对每一列的所有元素进行softmax运算，并使得每一列所有元素和为1。
        #（2）dim=1：对每一行的所有元素进行softmax运算，并使得每一行所有元素和为1。

        return outputs.detach()

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


class ClassifyData(data.Dataset):
    def __init__(self,data_set_count=0, train_data=None, is_training=None,embed_dim=0):
        super(ClassifyData, self).__init__() 
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.features_fill = train_data 
        self.embed_dim = embed_dim
    def __len__(self):  
        return self.data_set_count#return self.num_ng*len(self.train_dict)
          
    def __getitem__(self, idx):
        features = self.features_fill
        feature_user = features[idx][:self.embed_dim ]
        label_user = features[idx][self.embed_dim:]
        label_user = label_user.astype(np.int)
        return feature_user, label_user


def clf_age_all_pre(model_id,epoch_run,users_embs,factor_num,device):
    # #writer = SummaryWriter("logs")
    # seed = 100
    # np.random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # torch.manual_seed(seed)
    batch_size=2048*100  
    dataset_base_path='../../data/ml1m'

    epoch_id='clf_age/'+str(epoch_run) 
    #print(model_id,epoch_id)
    dataset='movieLens-1M'
    device=torch.device(device)
    users_features = np.load('./data/ml-1m/users_features_list.npy',allow_pickle=True)
    #6040*30的矩阵，one-hot编码的用户特征因为21+7+2=30

    users_features=users_features.astype(np.float32)
    users_features_age_oh=users_features[:,2:9]#取第二列到第九列
    users_features_age=[np.where(r==1)[0][0] for r in users_features_age_oh]
    #找到每一行1所在位置的索引
    users_features_age=np.array(users_features_age).astype(np.float32)#list转为array
    users_features_age = (users_features_age).reshape(-1,1)#转为列向量
    users_embs = users_embs.astype(np.float32)
    users_embs_cat_att = np.concatenate((users_embs, users_features_age), axis=-1)
    #矩阵拼接函数，将users_features_age拼在users_embs的最后一列变成6040*65的矩阵
    """
    axis=0：在第一维操作
    axis=1：在第二维操作
    axis=-1：在最后一维操作
    """
    np.random.shuffle(users_embs_cat_att)#按行打乱
    train_data_all = users_embs_cat_att[:-1000][:]#去掉后1000行，-1000表示从下往上数
    training_count=len(train_data_all)#5040个训练样本
    test_data_all = users_embs_cat_att[-1000:][:]
    testing_count=len(test_data_all)#1000个测试样本
     
    train_dataset = ClassifyData(
            data_set_count=training_count, train_data=train_data_all,is_training = True,embed_dim=factor_num)
    #处理后这个数据集将前64列作为输入，最后一列作为label
    train_loader = DataLoader(train_dataset,
            batch_size=training_count, shuffle=True, num_workers=2)#num_workers原本是2

    testing_dataset_loss = ClassifyData(
            data_set_count=testing_count, train_data=test_data_all,is_training = False,embed_dim=factor_num)
    testing_loader = DataLoader(testing_dataset_loss,
            batch_size=testing_count, shuffle=False, num_workers=0)


    ######################################################## TRAINING #####################################
    # print('--------training processing-------')
    count, best_hr = 0, 0 
    model = Classifier(embed_dim=factor_num, out_dim=7)
    model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#原来是#, betas=(0.5, 0.99))
 
    current_loss=0
    flag_stop=-1
    res_center=''
    res_p=[]
    res_r=[]
    for epoch in range(230):
        model.train()  
        start_time = time.time() 
        # print('train data of ng_sample is  end')
        train_loss_sum=[]
        train_loss_bpr=[]  
        for user_features, user_labels in train_loader:
            # pdb.set_trace()
            user_features = user_features.to(device)
            user_labels = user_labels.to(device)
            loss_get = model(user_features,user_labels)  
            optimizer.zero_grad()
            loss_get.backward()
            optimizer.step()  
            count += 1
            train_loss_sum.append(loss_get.item()) 

        elapsed_time = time.time() - start_time
        train_loss=round(np.mean(train_loss_sum),4)#
        #writer.add_scalar("loss", train_loss, epoch)
        str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+' train loss:'+str(train_loss) 
        if train_loss>current_loss:
            flag_stop+=1
        current_loss=train_loss

        model.eval()
        auc_test_all=[]
        for user_features, user_labels in testing_loader:
            user_features = user_features.to(device)
            user_labels = user_labels.numpy()
            get_scores = model.prediction(user_features)#用户embedding传入网络得到预测结果
            pre_scores=get_scores.cpu().numpy() 

            y = (user_labels).reshape(-1)
            pred = np.argmax(pre_scores,axis=-1)
            f1_macro = f1_score(y, pred, average='macro')
            f1_micro = f1_score(y, pred, average='micro')

            p_one = precision_score(y, pred, average='micro')
            r_one = recall_score(y, pred, average='micro')

            str_f1='age ,f1_macro:'+str(round(f1_macro,4))+'  f1_micro:'+str(round(f1_micro,4))#+' auc:'+str(round 
            if flag_stop>=2:
                # print("epoch:"+str(epoch)+str_f1)
                res_center+=str(round(f1_micro,4))+' '
                res_p.append(p_one)
                res_r.append(r_one)
    #return p_one,r_one,res_p,res_r
        #str_print_evl=str_print_train+" epoch:"+str(epoch)+str_f1
        #print(str_print_evl)
        #writer.add_scalar("f1", f1_micro, epoch)
        if flag_stop == 4:#原来是4
            return p_one, r_one, res_p, res_r
    if len(res_p)==0:
        res_p.append(p_one)
        res_r.append(r_one)
    return p_one, r_one, res_p, res_r
    







