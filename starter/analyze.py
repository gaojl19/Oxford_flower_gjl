from ast import parse
import jittor as jt
from jittor import nn
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR
from matplotlib.pyplot import plot

import datasets
from datasets.dataloader import *

import models
from models.model import *
import models.VAE as VAE
import models.SimCLR as SimCLR
import tools
from tools.train import train_one_epoch
from tools.test import *

import os
import json

jt.flags.use_cuda = 1
jt.set_global_seed(648)


model_dict = {
        'VAE':
        {
            'VIT': '/root/gjl/log/ViT_vae_lr0.001/model_best.pkl',
            'ConvMixer': '/root/gjl/log/Conv_vae_lr0.001/model_best.pkl',
            'MLPMixer': '/root/gjl/log/MLP_vae_lr0.001/model_best.pkl'
        },
        'SimCLR':
            {
                'VIT': '/root/gjl/log/ViT_simclr_lr0.001/model_best.pkl',
                'ConvMixer': '/root/gjl/log/Conv_simclr_lr0.001/model_best.pkl',
                'MLPMixer': '/root/gjl/log/MLP_simclr_lr0.001/model_best.pkl'
            }

    }

# 加载模型
def build_model(model_type, pretrain_type):
    if model_type == "ConvMixer":
        if pretrain_type == "simclr":
            model = SimCLR.ConvMixer_768_32_clr()
        else:
            model = VAE.ConvMixer_768_32_vae()

    elif model_type == "MLPMixer":
        if pretrain_type == "simclr":
            model = SimCLR.MLPMixer_S_16_clr()
        else:
            model = VAE.MLPMixer_S_16_vae()

    elif model_type == "VIT":
        if pretrain_type == "simclr":
            model = SimCLR.vit_small_patch16_224_clr()
        else:
            model = VAE.vit_small_patch16_224_vae()

    else:
        raise NotImplementedError("model type {model_type} is not supported right now\n")
    
    return model


def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="choose the model architecture: ConvMixer, MLPMixer, VIT")  # relative to where you're running this script from
    parser.add_argument('--epoch_num', type=int, required=True, help="how many epochs for training")
    parser.add_argument('--learning_rate', type=float, default=0.003, help="learning rate for training")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay rate for training")
    parser.add_argument('--save_model', type=bool, default=False, help="whether save the best model while training")
    parser.add_argument('--save_dir', type=str, default="./", help="where to save the model from training")
    parser.add_argument('--load_from_checkpoint', type=str, default=None, help="if not empty, load from path")
    parser.add_argument('--test', type=bool, default=False, help="whether to use test mode")
    parser.add_argument('--save_result_dir', type=str, default=None, help="where to save")
    parser.add_argument('--pretrain_type', type=str, default=None, help="simclr or vae")



    args = parser.parse_args()
    # convert args to dictionary
    args = vars(args)

    # alter save dir
    if args["test"]:
        save_dir = args["load_from_checkpoint"]

    else:
        save_dir = args["save_dir"]
        save_dir += "_lr" + str(args["learning_rate"]) + "/"

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    args["save_dir"] = save_dir
    
    return args


def plot_loss_curve(curve, plot_prefix):
    
    import pandas as pd
    import seaborn as sns

    iteration = range(1, len(curve)+1)
    data = pd.DataFrame(curve, iteration)
    ax=sns.lineplot(data=data)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Train Loss Curve")
    
    fig = ax.get_figure()
    print("saving to:", plot_prefix)
    fig.savefig(plot_prefix + "loss_curve.png")
    
    fig.clf()


def plot_acc_curve(train_acc, valid_acc, plot_prefix):
    
    import pandas as pd
    import seaborn as sns

    assert len(train_acc)==len(valid_acc)
    iteration = range(1, len(train_acc)+1)
    df = {
        "train_acc": train_acc,
        "valid_acc": valid_acc
    }
    data = pd.DataFrame(data=df, index=iteration)
    ax=sns.lineplot(data=data)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve")
    ax.set(ylim=(-0.1, 1.1))
    
    fig = ax.get_figure()
    fig.savefig(plot_prefix + "accuracy_curve.png")
    
    fig.clf()




def test(args, dataloader):
    
    model = build_model(args["model_name"], args["pretrain_type"])
    # 从checkpoint中加载
    if args["load_from_checkpoint"] != None:
        model.load(args["load_from_checkpoint"] + "model_best.pkl")

    acc = test_one_epoch(model, dataloader.testdataset, args["save_result_dir"] )
    
    # 打印效果
    print("*****************************************************")
    print(f" Test Accuracy: {acc} \n")
    with open(args["save_dir"]+"result.txt", "w") as fout:
        fout.write(f" Test Accuracy: {acc} \n")

    fout.close()



if __name__ == "__main__":
    
    # generate args
    # args = parse_arg()
    # print(args)
    
    # dataloader = Dataset("./data")
    # test(args, dataloader)

    import json
    file = "../cyl/log/ConvMixer_lr0.001/log.json"
    with open(file, 'r') as fin:
        content = json.load(fin)
        loss_ = content["loss"]
        train_acc = content["train_acc"]
        valid_acc = content["valid_acc"]
    fin.close()
    loss = []

    for l in loss_:
        if l>0:
            loss.append(l)
    plot_loss_curve(loss, "../cyl/log/ConvMixer_lr0.001/")
    plot_acc_curve(train_acc, valid_acc, "../cyl/log/ConvMixer_lr0.001/")
    

    
