from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import MultiKR
from train import train_model,TrainSet
from utils import load_rating,load_kg
import argparse

def main(args):
    
    device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

    n_user,n_item,train_rec,eval_rec,test_rec =load_rating()
    n_entity,n_relation,kg = load_kg()

    kg_data = (kg[:,0],kg[:,1],kg[:,2])
    rec_data = (train_rec[:, 0], train_rec[:, 1], train_rec[:, 2])
    rec_val = (eval_rec[:, 0], eval_rec[:, 1], eval_rec[:, 2])
    train_data_kg = TrainSet(kg_data)
    train_loader_kg = DataLoader(train_data_kg, batch_size=args.batch_size, shuffle=args.shuffle_train)
    train_data_rec = TrainSet(rec_data)
    eval_data_rec = TrainSet(rec_val)

    train_loader_rec = DataLoader(train_data_rec, batch_size=args.batch_size, shuffle=args.shuffle_train)
    eval_loader_rec = DataLoader(eval_data_rec, batch_size=args.batch_size, shuffle=args.shuffle_test)
    model = MultiKR(n_user + 1, n_item + 1, n_entity + 1, n_relation + 1, n_layer=args.n_layer,
                    embed_dim=args.batch_size,
                    hidden_layers=args.hidden_layers,
                    dropouts=args.dropouts, output_rec=args.output_rec)
    model = model.to(device)
    optimizer =torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay,lr=args.lr)
    loss_fuction=nn.BCEWithLogitsLoss()
    epochs = args.epochs
    # print('='*10+str(type(train_loader_kg))+"="*10)
    train_model(model,train_loader_rec,train_loader_kg,eval_loader_rec,optimizer,loss_fuction,epochs)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="mkr model arguments")
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle_train", type=bool, default=True)
    parser.add_argument("--shuffle_test", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--output_rec", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_layers", nargs='+', type=int, default=[64, 64])
    parser.add_argument("--dropouts", nargs='+', type=float, default=[0.5, 0.5])
    args = parser.parse_args()
    main(args)