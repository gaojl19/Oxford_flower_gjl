import os

import tqdm

import models.VAE as VAE
import models.transformer as t
import models.SimCLR as SimCLR
import cv2
import colorsys
# from utils import *

import argparse
import numpy as np
import sklearn.decomposition
import sklearn.manifold


import matplotlib.pyplot as plt

import nltk

import jittor as jt
jt.flags.use_cuda = 1


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--method', type=str, default='')
    return parser.parse_args()


def closest_string(target, selections):
    target = target.lower()
    r = ''
    x = 1e100
    for selection in selections:
        y = nltk.edit_distance(target, selection.lower())
        if x > y:
            x = y
            r = selection
    return r


def build_model(method, model_type):
    if method == 'VAE':
        if model_type == "ConvMixer":
            model = VAE.ConvMixer_768_32_vae()
        elif model_type == "MLPMixer":
            model = VAE.MLPMixer_S_16_vae()
        elif model_type == "VIT":
            model = VAE.vit_small_patch16_224_vae()
        else:
            raise NotImplementedError("model type {model_type} is not supported right now\n")
    else:
        if model_type == "ConvMixer":
            model = SimCLR.ConvMixer_768_32_clr()
        elif model_type == "MLPMixer":
            model = SimCLR.MLPMixer_S_16_clr()
        elif model_type == "VIT":
            model = SimCLR.vit_small_patch16_224_clr()
        else:
            raise NotImplementedError("model type {model_type} is not supported right now\n")
    return model

if (__name__ == '__main__'):
    args = get_arguments()
    method = closest_string(args.method, ['VAE', 'SimCLR'])
    modelname = closest_string(args.model, ['ConvMixer', 'MLPMixer', 'VIT'])

    # img_paths = args.img_dir.split()
    task = 3
    if task == 3:
        img_dirs = ['data/test']

        image_paths = []

        for img_dir in img_dirs:
            img_dir = '/root/gjl/' + img_dir
            classes_ = os.listdir(img_dir)
            for c in classes_:
                if int(c) > 30:
                    continue
                p = img_dir + '/' + c
                imgs = os.listdir(p)
                for img in imgs:
                    path = p + '/' + img
                    image_paths.append((path, c))

        inputs = []
        classes = []
        for path, c in tqdm.tqdm(image_paths[:]):
            input_image = cv2.imread(path)

            if input_image is not None:
                input_image = cv2.resize(input_image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = input_image.transpose(2, 0, 1).astype(float) / 255
                input_ = jt.array(input_image.reshape((1, *input_image.shape)))
                # print(input_image.shape, input_.shape)
                inputs.append((input_, c))
                classes.append(c)

        classes_set = set(classes)
        print(f"{len(classes)} images, {len(classes_set)} classes")
        colors = {}
        N = len(classes_set)


        def get_color(w):
            h = w * 2 / N * 360
            s = w / N * 51 + 50
            v = w / N * 43 + 58
            return colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)

        for cls in classes_set:
            w = int(cls)

            colors[cls] = get_color(w)

    else:
        img_paths = [
            'data/train/0/image_06736.jpg@black',  # white
            'data/train/33/image_06946.jpg',
            'data/valid/6/image_07210.jpg',
            'data/train/41/image_05731.jpg',

            'data/train/13/image_06066.jpg@pink',  # pink
            'data/train/33/image_06931.jpg',

            'data/test/55/image_02777.jpg@red',  # red
            'data/test/57/image_02681.jpg',
            'data/test/5/image_07172.jpg',
            'data/test/99/image_07894.jpg',

            'data/test/44/image_07145.jpg@purple',  # purple
            'data/train/24/image_06601.jpg',
            'data/train/24/image_06592.jpg',
            'data/train/24/image_06582.jpg',
            'data/train/44/image_07131.jpg',

            'data/train/5/image_07195.jpg@yellow',  # yellow'
            'data/train/15/image_06688.jpg',
            'data/valid/46/image_04988.jpg',
            'data/train/4/image_05167.jpg',
            'data/train/4/image_05182.jpg',
            'data/train/4/image_05186.jpg',
            'data/train/4/image_05187.jpg',

            'data/test/18/image_06168.jpg@blue',  # blue
            'data/valid/18/image_06185.jpg',
            'data/train/27/image_05261.jpg',
            'data/train/24/image_06593.jpg',
        ]
        inputs = []
        colors = []
        classes = []
        last_color = 'black'
        for img_path in img_paths:
            img_path, *_ = img_path.split('@')
            if len(_) > 0:
                last_color = _[0]

            colors.append(last_color)
            if img_path.startswith('data'):
                img_path = '/root/gjl/' + img_path

            input_image = cv2.imread(img_path)

            if input_image is not None:
                c = len(colors) - 1
                input_image = cv2.resize(input_image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = input_image.transpose(2, 0, 1).astype(float) / 255
                input_ = jt.array(input_image.reshape((1, *input_image.shape)))
                # print(input_image.shape, input_.shape)
                inputs.append((input_, c))
                classes.append(c)

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
    methods = []
    modelnames = []
    if task == 3:
        for method in model_dict:
            for modelname in model_dict[method]:
                methods.append(method)
                modelnames.append(modelname)
    else:
        methods.append(method)
        modelnames.append(modelname)

    for method, modelname in zip(methods, modelnames):
        print(method, modelname)
        model = build_model(method, modelname)
        # model.load(model_dict[method][modelname])
        model.eval()

        embs = []
        for input_, c in tqdm.tqdm(inputs):
            input_ = input_.float32()  ### HERE!
            emb = model.feature(input_)
            emb = emb.reshape(-1)
            z = emb.numpy()
            # print("AFTER input: ", z.shape)

            embs.append(z)

        # vgg = models.vgg16(pretrained=True)
        # vgg = MLPMixer_S_16()
        # vgg.load('../../model_best.pkl')
        # vgg = vgg
        #
        # embs = []
        # for input_ in inputs:
        #     pred = vgg(input_)
        #     emb = vgg.get_feature()
        #     z = emb.numpy().reshape(-1)
        #     print(z.shape)
        #     embs.append(z)
        #
        pca = sklearn.decomposition.PCA(2)
        es = pca.fit_transform(embs)
        color_array = [colors[classes[i]] for i in range(len(inputs))]
        # print(es)
        plt.figure()
        plt.scatter(es[:, 0], es[:, 1], c=color_array)
        plt.savefig(f"plt{task}_{method}_{modelname}_PCA.png")
        # for i, (x, y) in enumerate(es):
        #     plt.plot(x, y, '.', color=colors[classes[i]])
        # plt.savefig(f"plt_{method}_{modelname}_pca.png")
        plt.close()
        tsne = sklearn.manifold.TSNE()
        X_embedded = tsne.fit_transform(embs)
        plt.figure()
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color_array)
        plt.savefig(f"plt{task}_{method}_{modelname}_TSNE.png")
        plt.close()