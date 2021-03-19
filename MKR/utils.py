import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random

def linear_layer(input,output,dropout=0):
    return nn.Sequential(
        nn.Linear(input,output),
        nn.LeakyReLU(),
        nn.Dropout(dropout)

    )
def daraset_split(rating_np):
    pass
    eval_ratio =0.2
    test_ratio =0.2
    n_ratings=rating_np.shape[0]
    train_vali_data,test_data=train_test_split(rating_np,test_size=test_ratio,random_state=2021,shuffle=True)
    train_data,eval_data = train_test_split(train_vali_data,test_size=0.25,random_state=2021,shuffle =True)
    return train_data,eval_data,test_data

def load_rating():
    print('reading rating file ...')

    # reading rating file
    rating_file = './MKR-data/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    n_user = len(set(rating_np[:,0]))
    n_item = len(set(rating_np[:,1]))
    train_data,eval_data,test_data = daraset_split(rating_np)
    return n_user,n_item,train_data,eval_data,test_data

def load_kg():
    print('reading KG file ...')

    # reading kg file
    kg_file = './MKR-data/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg = np.load(kg_file + '.npy')
    else:
        kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg)
    
    n_entity= len(set(kg[:,0])|set(kg[:,2]))
    n_relation = len(set(kg[:,1]))
    return n_entity,n_relation,kg


def multi_loss(pred, target, types, loss_function):
    if types == "rec":
        loss = loss_function(pred, target)
        return loss
    else:

        loss = torch.sigmoid(torch.sum(pred * target))
        return loss
    
    


# if __name__=='__main__':
    # name ={
    #      1:'曾瑾',
    #      2:'蔡文增',
    #      3:'刘泽洋',
    #      4:'王业超',
    #      5:'钟胜杰',
    #      6:'周蓓',
    #      7:'朱浩峰',
    #      8:'陈妙慧' }
    # num = random.randint(1,8)

    # print(name[num])



