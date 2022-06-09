from re import A
from jittor import argmax
import jittor
import numpy as np
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import random
import colorsys
from sklearn.preprocessing import normalize
import cv2


# import jittor

def plot_feature(feature_dict, labels, tag, save_dir):
    '''
        plot t-sne of features of specific tasks
    '''
    print("plot feature!")
    X = []
    y = []
    for l in labels:
        for f in feature_dict[l]:
            X.append(f.squeeze(0))
            y.append("class "+ str(l))
    
    X = np.array(jittor.stack(X))
    
    # using scikit-learn package
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X) 
    palette = sns.color_palette("bright", len(set(y)))
    plt.figure(figsize=(8,8))
    sns_plot = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)
    fig = sns_plot.get_figure()

    fig.savefig(save_dir + tag + "_tsne.png")


def avg_attn_distance(attn_weights, sample_n, tag):

    df = {
        "head": [],
        "Network depth": [],
        "attention distance": []
    }
    depth = 0
    for w in tqdm(attn_weights):
        w = w.squeeze(0)
        head_num = w.shape[0]
        w = w[:, 0, 1:].reshape(-1, 14, 14)
        row = w.shape[1]
        col = w.shape[2]
        # print(head_num, row, col)

        for head in range(head_num):
            dist = 0
            cnt = 0
            for i in range(row):
                for j in range(col):
                    v1 = np.array([i, j])
                    for _i in range(row):
                        for _j in range(col):
                            v2 = np.array([_i, _j])  
                            dist += float(w[head][_i][_j]) * np.sqrt(np.sum(np.square(v1-v2))) 

            df["attention distance"].append(dist)
            df["Network depth"].append(depth)
            df["head"].append(head)
        depth += 1

    # plot
    data=pd.DataFrame(df)
    plt.figure(figsize=(8,8))
    palette = sns.color_palette("bright", head_num)
    sns_plot = sns.scatterplot(x="Network depth", y="attention distance", hue="head", data=data, s=20, palette=palette)
    fig = sns_plot.get_figure()
    fig.savefig("./output/attention_distances/" + tag + ".png")


def plot_attention_map(im, att_mat, tag):
    att_mat = jittor.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = jittor.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = jittor.init.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = jittor.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = jittor.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = np.array(v[0, 1:].reshape(grid_size, grid_size))
    im = im.squeeze(0)
    im = np.array(im).swapaxes(0,1).swapaxes(1,2)
    mask = cv2.resize(mask/mask.max(), (im.shape[0], im.shape[1]))[..., np.newaxis]
    result = (mask * np.array(im)).astype("uint8")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)
    plt.savefig('./output/attention_maps/' + tag + '_attention.png')


    print("Prediction Label and Attention Map!\n")
    

def avg_attn_rollout(image, attn_weights, tag):
    '''
        do attention rollouts,
    '''

    attn_map = np.ones([attn_weights[0].shape[2], attn_weights[0].shape[3]])
    # attn_map = normalize(attn_map, axis=0, norm='max')

    i=0
    for w in attn_weights:
        i += 1
        w = np.array(w.squeeze(0))
        attn_map *= np.mean(np.array(w), axis=0)

    # print(attn_map)
    # attn_map = np.tile(attn_map[:,:,None], 3)
    attn_map = attn_map[0, 1:].reshape(14,14)
    mask = cv2.resize(attn_map/attn_map.max()*255, (224, 224))[..., np.newaxis]
    plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig('./output/attention_maps/' + tag + '_attention.png')

    image = image.squeeze(0)
    
    img = np.array(image).swapaxes(0, 1).swapaxes(1, 2)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('./output/attention_maps/' + tag + '_origin.png')

    result = (mask * img).astype("uint8")
    plt.imshow(result)
    plt.savefig('./output/attention_maps/' + tag + '.png')
    


def plot_attention_distance(attn_dict, class_label, num=1):
    
    total_sample = len(attn_dict[class_label])
    chosen_sample = random.sample(range(total_sample), num)

    for i in chosen_sample:
        avg_attn_distance(attn_dict[class_label][i], 1, "class_" + str(class_label) + "_" + str(i))


def plot_attention_rollout(image_dict, attn_dict, class_label, num=1):

    total_sample = len(attn_dict[class_label])
    chosen_sample = random.sample(range(total_sample), num)

    for i in chosen_sample:
        avg_attn_rollout(image_dict[class_label][i], attn_dict[class_label][i], "class_" + str(class_label) + "_" + str(i))
        # plot_attention_map(image_dict[class_label][i], attn_dict[class_label][i], "class_" + str(class_label) + "_" + str(i))


def calculate_prec_recall(matrix, total_labels, model_name, plot=True):
    FP = matrix.sum(axis = 0) # 列  prec
    FN = matrix.sum(axis = 1) # recall

    for i in range(0,total_labels):
        TP = matrix[i][i]
        FP[i] = TP/(FP[i])
        FN[i] = TP/(FN[i])

    worst_fp = np.argmin(FP)
    worst_fn = np.argmin(FN)
    best_fp = np.argmax(FP)
    best_fn = np.argmax(FN)
    
    print("*****************prec*****************")
    print(FP[55])
    print(FP[3])
    print("worst fp: ", worst_fp, FP[worst_fp])
    print(matrix[...,worst_fp].argsort()[::-1][0:5])

    print("best fp: ", best_fp, FP[best_fp])
    print(matrix[...,best_fp].argsort()[::-1][0:5])
    # print(matrix[...,worst_fp])


    print("*****************recall***************")
    print(FN[62])
    print(FN[42])
    print("worst fn: ", worst_fn, FN[worst_fn])
    print(matrix[worst_fn].argsort()[::-1][0:5])

    print("best fn: ", best_fn, FN[best_fn])
    print(matrix[...,best_fn].argsort()[::-1][0:5])

    # print(matrix[worst_fn])
    print("**************************************")

    if plot:
        ax = plt.subplots()
        plt.bar(np.arange(total_labels),FP)
        plt.xlabel("label")
        plt.title("prec "+model_name)
        plt.savefig("./output/prec_"+model_name+".png")
        plt.show()
        plt.clf()

        ax = plt.subplots()
        plt.bar(np.arange(total_labels),FN)
        plt.xlabel("label")
        plt.title("recall " + model_name)
        plt.savefig("./output/recall_"+model_name+".png")
        plt.show()
        plt.clf()


def plot_heatmap(matrix, model_name):
    np.save("./log/matrix/"+model_name,matrix)
    ax = sns.heatmap(matrix,annot=True,square=True,cmap = 'YlGnBu')
    ax.set_xlabel("pred",fontsize=40)
    ax.set_ylabel("true",fontsize=40)
    ax.set_title (model_name,fontsize =50)
    fig = ax.get_figure()
    sns.plt.show()
    fig.savefig("./output/heatmap_"+model_name+".png")
    fig.clf()


def rgb2gray(rgb):
    r, g, b=rgb[: ,0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray=0.2989*r + 0.5870*g + 0.1140*b
    return gray


def plot_vae(images, output, save_dir):
    idx = random.sample(range(len(images)), 5)
    for i in idx:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

        ax1.set_title('Original')
        ax2.set_title('Output')
        avg_im = images[i].squeeze(0).mean(0)
        # im = np.array(images[i]).squeeze(0).swapaxes(0,1).swapaxes(1,2)
        im = np.array(avg_im.repeat(3,1,1)).swapaxes(0,1).swapaxes(1,2)

        out = output[i].reshape(1, 224, 224).repeat(3,1,1)
        # print(out.shape)
        out = np.array(out).swapaxes(0,1).swapaxes(1,2)
        # print(out)
        # print(im)
        _ = ax1.imshow(im)
        _ = ax2.imshow(out)
        plt.savefig(save_dir + str(i+1) + '_vae.png')

def test_one_epoch(model, test_loader, save_dir):

    modelname = "VIT"
    idx2label = test_loader.classes
    model.eval()
    total_acc = 0
    total_num = 0

    feature_dict = {}
    attn_dict = {}
    image_dict = {}

    total_labels = 102
    matrix = np.zeros((total_labels,total_labels)) # [实际][预测]
    for i in range(total_labels): # label -> [feature vectors]
        feature_dict[i] = []
        attn_dict[i] = []
        image_dict[i] = []

    plt.figure(figsize=(30, 30))

    # pbar = tqdm(test_loader, desc=f'Testing')
    pbar = tqdm(test_loader, desc=f'Testing')
    image_list = []
    output_list = []
    for i, (images, labels) in enumerate(pbar):
        
        # convert image to gray images
        # images = rgb2gray(images)
        # print(images.shape)

        # add feature vector return
        # output, feature, attn_weights = model(images, True, True)
        output = model(images)
        image_list.append(images)
        output_list.append(output)
        
        if len(output_list) > 100:
            break
        # pred = np.argmax(output.data, axis=1)
        # acc = np.sum(pred == labels.data)
        # total_acc += acc
        # total_num += labels.shape[0]

        # pbar.set_description(f'Test acc={total_acc / total_num:.2f}')
        
        # _pred = idx2label[pred[0]]
        _label = idx2label[labels.data[0]]
        # matrix[(int)(_label)][(int)(_pred)] += 1 # 混淆矩阵

        # if (int)(_label) in [3,87,74,75,93,50,55,57,43,17,82,42,5,84,89,73,10,62,64,70,45,4]:
        #     feature_dict[(int)(_label)].append(feature[:, 0])

    plot_vae(image_list, output_list, save_dir)
    
    # calculate_prec_recall(matrix=matrix, total_labels=total_labels, model_name=modelname, plot=False)
    # plot_heatmap(matrix=matrix, model_name=modelname)


    # plot features of different task groups
    # plot_feature(feature_dict, [3,87,74,75,93,50], tag="FP_worst", save_dir=save_dir)
    # plot_feature(feature_dict, [55,57,43,17,82], tag="FP_best", save_dir=save_dir)
    # plot_feature(feature_dict, [42,5,84,89,73,10], tag="FN_worst", save_dir=save_dir)
    # plot_feature(feature_dict, [62,64,70,45,4], tag="FN_best", save_dir=save_dir)

    # # plot attention distance for class 3, 55, 42, 62
    # plot_attention_distance(attn_dict=attn_dict, class_label=3, num=1)
    # plot_attention_distance(attn_dict=attn_dict, class_label=55, num=1)
    # plot_attention_distance(attn_dict=attn_dict, class_label=42, num=1)
    # plot_attention_distance(attn_dict=attn_dict, class_label=62, num=1)

    # plot attention rollouts for class 3, 55, 42, 62
    # plot_attention_rollout(image_dict=image_dict, attn_dict=attn_dict, class_label=3, num=2)
    # plot_attention_rollout(image_dict=image_dict, attn_dict=attn_dict, class_label=55, num=2)
    # plot_attention_rollout(image_dict=image_dict, attn_dict=attn_dict, class_label=42, num=2)
    # plot_attention_rollout(image_dict=image_dict, attn_dict=attn_dict, class_label=62, num=2)
    
    acc=None
    if total_num>0:
        acc = total_acc / total_num
    return acc