from ast import parse
import jittor as jt
from jittor import nn
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR
from matplotlib.pyplot import plot
import sys
sys.path.append(".") 

import datasets
from datasets.dataloader import *

import models
from models.model import *
from models.transformer import *

import tools
from tools.train import *

import os
import json

# jt.flags.use_cuda = args["cuda"]
jt.set_global_seed(648)

# 加载模型
def build_model(model_type):
    if model_type == "ConvMixer":
        model = ConvMixer_768_32()
    elif model_type == "MLPMixer":
        model = MLPMixer_S_16()
    elif model_type == "VIT":
        model = vit_small_patch16_224()
    else:
        raise NotImplementedError("model type {model_type} is not supported right now\n")
    
    return model


def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help="choose the model architecture: ConvMixer, MLPMixer, VIT")  # relative to where you're running this script from
    parser.add_argument('--model_dir', type=str, default=None, help="Pre-trained model address")
    parser.add_argument('--cuda', type=int, required=True, help="CUDA num")
    parser.add_argument('--epoch_num', type=int, required=True, help="how many epochs for training")
    parser.add_argument('--learning_rate', type=float, default=0.003, help="learning rate for training")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay rate for training")
    parser.add_argument('--save_model', type=bool, default=False, help="whether save the best model while training")
    parser.add_argument('--save_dir', type=str, default="./", help="where to save the model from training")
    parser.add_argument('--load_from_checkpoint', type=str, default=None, help="if not empty, load from path")
    parser.add_argument('--test', type=bool, default=False, help="whether to use test mode")
    parser.add_argument('--data_dir', type=str, default="./data", help="where to load data for train, valid and test")



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


def train(args, dataloader):

    model = build_model(args["model_name"])

    # load from pretrain
    if args["model_dir"]!=None:
        model.load(args["model_dir"])

    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = MultiStepLR(optimizer, milestones=[40, 80, 160, 240], gamma=0.2) # learning rate decay
    # scheduler = CosineAnnealingLR(optimizer, 15, 1e-5)

    # 从checkpoint中加载
    if args["load_from_checkpoint"] != None:
        model.load(args["load_from_checkpoint"])

    # 训练参数
    epochs = args["epoch_num"]
    best_acc = 0.0
    best_epoch = 0

    # 可视化loss和accuracy
    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, dataloader.traindataset, criterion, optimizer, epoch, 1, scheduler)
        acc = valid_one_epoch(model, dataloader.validdataset, epoch)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

            # 在这里保存模型
            if args["save_model"]:
                model.save(args["save_dir"] + f'model_best.pkl')
        
        # 记录当前epoch的结果
        train_loss_list += np.array(train_loss).tolist()
        train_acc_list.append(float(train_acc))
        valid_acc_list.append(float(acc))

        json_dict = {"loss": train_loss_list, "train_acc": train_acc_list, "valid_acc": valid_acc_list}
        weight_json = json.dumps(json_dict, sort_keys=False, indent=4)
        f = open(args["save_dir"] +  "log.json", 'w')
        f.write(weight_json)
        f.close()
        print(train_loss)

    # 打印效果
    print("*****************************************************")
    print(f" best_acc: {best_acc},  best_epoch: {best_epoch} \n")
    with open(args["save_dir"]+"test_result.txt", "w") as fout:
        fout.write(f" best_acc: {best_acc},  best_epoch: {best_epoch} \n")

    fout.close()

    # 可视化loss和accuracy
    plot_loss_curve(train_loss_list, args["save_dir"])
    plot_acc_curve(train_acc=train_acc_list, valid_acc=valid_acc_list, plot_prefix=args["save_dir"])


def test(args, dataloader):
    
    model = build_model(args["model_name"])
    model.load(args["model_dir"])
    # 从checkpoint中加载
    if args["load_from_checkpoint"] != None:
        model.load(args["load_from_checkpoint"] + "model_best.pkl")

    acc = test_one_epoch(model, dataloader.testdataset)
    
    # 打印效果
    print("*****************************************************")
    print(f" Test Accuracy: {acc} \n")
    with open(args["save_dir"]+"result.txt", "w") as fout:
        fout.write(f" Test Accuracy: {acc} \n")

    fout.close()



if __name__ == "__main__":
    # generate args
    args = parse_arg()
    print(args)

    dataloader = Dataset(dataset_path=args["data_dir"])

    if args["test"]:
        test(args, dataloader)
    else:
        train(args, dataloader)
    

    
