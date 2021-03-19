import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import linear_layer
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
class MultiKR(nn.Module):
    def __init__(self,user_num,item_num,entity_num,relation_num,n_layer,embed_dim,hidden_layers,dropouts,output_rec):

        super(MultiKR,self).__init__()
        self.user_embed = nn.Embedding(user_num,embed_dim)
        self.item_embed = nn.Embedding(item_num,embed_dim)
        self.entity_embed = nn.Embedding(entity_num,embed_dim)
        self.relation_embed = nn.Embedding(relation_num,embed_dim)
        self.n_layer = n_layer
        self.w_vv = torch.rand((embed_dim,1),requires_grad=True)
        self.w_ev = torch.rand((embed_dim,1),requires_grad=True)
        self.w_ve = torch.rand((embed_dim,1),requires_grad=True)
        self.w_ee = torch.rand((embed_dim,1),requires_grad=True)
        self.bais_v = torch.rand(1,requires_grad=True)
        self.bais_e = torch.rand(1,requires_grad=True)

        # self.user_embed = self.user_embed.to(device)
        # self.item_embed = self.item_embed.to(device)
        # self.entity_embed = self.entity_embed.to(device)
        # self.relation_embed = self.relation_embed.to(device)
        self.w_vv = self.w_vv.to(device)
        self.w_ev = self.w_ev.to(device)
        self.w_ve = self.w_ve.to(device)
        self.w_ee = self.w_ee.to(device)
        self.bais_v = self.bais_v.to(device)
        self.bais_e = self.bais_e.to(device)

        #mlp for low layer
        self.user_low_mlp_layer = linear_layer(embed_dim,embed_dim,dropout=0.5)
        self.relation_low_mlp_layer =linear_layer(embed_dim,embed_dim,dropout=0.5)
        # mlp for kge
        self.kg_layer = nn.Sequential()
        layers =[2*embed_dim]+hidden_layers
        for i in range(len(layers)-1):
            self.kg_layer.add_module(
                'kg_hidden_layer_{}'.format(i+1),
                linear_layer(layers[i],layers[i+1],dropouts[i])
            )
        self.kg_layer.add_module('kg_last_layer',linear_layer(layers[-1],embed_dim))
        # mlp for recommad
        self.rec_layer=nn.Sequential()
        layers=[2*embed_dim]+hidden_layers
        for i in range(len(layers)-1):
            self.rec_layer.add_module(
                'rec_hidden_layer_{}'.format(i+1),
                linear_layer(layers[i],layers[i+1],dropouts[i])
            )
        self.rec_layer.add_module('rec_last_layer',linear_layer(layers[-1],output_rec))
    
    def _init_weight(self):
        nn.init.xavier_uniform(self.user_embed.weight)
        nn.init.xavier_uniform(self.item_embed.weight)

    def cross_compress_unit(self,item_embed,head_embed):
        # batch_size * embed_dim * 1
        item_embed_reshape = item_embed.unsqueeze(-1)
        head_embed_reshape = head_embed.unsqueeze(-1)
        # batch_size * embed_dim * embed_dim
        c =item_embed_reshape *head_embed_reshape.permute((0,2,1))
        c_t = head_embed_reshape *item_embed_reshape.permute((0,2,1))
        
        item_embed_c = torch.matmul(c,self.w_vv).squeeze()+torch.matmul(c_t,self.w_ev).squeeze()+self.bais_v
        head_embed_c = torch.matmul(c,self.w_ve).squeeze()+torch.matmul(c_t,self.w_ee).squeeze()+self.bais_e
        return item_embed_c,head_embed_c


    def forward(self,data,train_type):
        # data = (x.detach().cpu().numpy() for x in data)
        # print('========',type(data[0]))
        if train_type =='rec':
            user_embed = self.user_embed(data[0].long())
            # print('==========',user_embed.device)
            item_embed =self.item_embed(data[1].long())
            # relation_embed = self.relation_embed(data[1].long())
            head_embed = self.entity_embed(data[1].long())
            rec_target =data[2].float()
            for i in range(self.n_layer):
                user_embed = self.user_low_mlp_layer(user_embed)
                item_embed,head_embed = self.cross_compress_unit(item_embed,head_embed)
            high_layer = torch.cat((user_embed,item_embed),dim=1)
            rec_out = self.rec_layer(high_layer)
            return rec_out.squeeze(),rec_target
        else:
            head_embed=self.entity_embed(data[0].long())
            item_embed = self.item_embed(data[0].long())
            relation_embed =self.relation_embed(data[1].long())
            tail_embed =self.entity_embed(data[2].long())
            for i in range(self.n_layer):
                item_embed,head_embed = self.cross_compress_unit(item_embed,head_embed)
                relation_embed = self.relation_low_mlp_layer(relation_embed)
            high_layer =torch.cat((head_embed,relation_embed),dim=1)
            tail_out = self.kg_layer(high_layer)
            return tail_out,tail_embed