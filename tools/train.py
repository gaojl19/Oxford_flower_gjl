from imp import new_module
from jittor import argmax
from jittor import nn
import jittor as jt
import numpy as np
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

KERNEL_DICT = {
    "gradient_laplacian_kernel" : np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
    "gaussian_smooth_kernel" : np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]),
    "vertical_kernel" : np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    "horizontal_gradient_kernel" : np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    "ruihua_kernel" : np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
    "soble_margin_kernel" : np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
}




def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler, contrast=False, kernel_type=None, vae=False):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for i, (images, labels) in enumerate(pbar):
        
        if contrast: # simslr
            # image [N, C, H, W] = [batch_size, 3, height, width]
            # choose a kernel
            kernel = KERNEL_DICT[kernel_type]
            conv = nn.Conv2d(3, 1, kernel_size=3, padding=1, groups=1, stride=1, bias=False)
            conv.weight.data = jt.Var([kernel, kernel, kernel]).float().unsqueeze(0)
        
            # get the image after being convolutioned by that kernel
            new_image = conv(images).detach().repeat(1,3,1,1)

            # plot the new image and old image
            # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

            # ax1.set_title('Original')
            # ax2.set_title('After convolution')
            # TAG = "6"
            # _ = ax1.imshow(np.array(images[1]).swapaxes(0,1).swapaxes(1,2))
            # _ = ax2.imshow(np.array(new_image[1]).swapaxes(0,1).swapaxes(1,2))
            # plt.savefig('./output/Simclr_visualization_' + TAG + '_1.png')

            # _ = ax1.imshow(np.array(images[5]).swapaxes(0,1).swapaxes(1,2))
            # _ = ax2.imshow(np.array(new_image[5]).swapaxes(0,1).swapaxes(1,2))
            # plt.savefig('./output/Simclr_visualization_' + TAG + '_2.png')

            # _ = ax1.imshow(np.array(images[12]).swapaxes(0,1).swapaxes(1,2))
            # _ = ax2.imshow(np.array(new_image[12]).swapaxes(0,1).swapaxes(1,2))
            # plt.savefig('./output/Simclr_visualization_' + TAG + '_3.png')

            # _ = ax1.imshow(np.array(images[8]).swapaxes(0,1).swapaxes(1,2))
            # _ = ax2.imshow(np.array(new_image[8]).swapaxes(0,1).swapaxes(1,2))
            # plt.savefig('./output/Simclr_visualization_' + TAG + '_4.png')


            # append to input list
            images = jt.concat([images,new_image])
            # print(images)
            output = model(images)
            loss = criterion(output)
            print(loss, images.shape[0])
            optimizer.backward(loss)

            # check gradient
            # for p in model.parameters():
            #     print(p.opt_grad(optimizer))

            # exit(0)

            if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
                optimizer.step(loss)
            losses.append(loss.data[0])

            pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f} ')

        elif vae:
            output = model(images)
            batch_size = int(images.shape[0])
            output = jt.reshape(output, (batch_size,224,224))
            images = images.mean(dim=1)
            loss = criterion(output, images)

            optimizer.backward(loss)
            if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
                optimizer.step(loss)

            losses.append(loss.data[0])
            pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f} ')


        else:
            output = model(images)
            loss = criterion(output, labels)
            # print(output.shape, labels.shape)
            # exit(0)

            optimizer.backward(loss)
            if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
                optimizer.step(loss)

            pred = np.argmax(output.data, axis=1)
            acc = np.sum(pred == labels.data)
            total_acc += acc
            total_num += labels.shape[0]
            losses.append(loss.data[0])

            pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f} '
                                f'acc={total_acc / total_num:.2f}')
    scheduler.step()

    if contrast or vae: # simclr or vae
        return losses

    return losses, total_acc / total_num
    
 
def valid_one_epoch(model, val_loader, epoch, contrast=False, kernel_type=None, criterion = None, vae=False):
    model.eval()
    total_acc = 0
    total_num = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    with jt.no_grad():

        for i, (images, labels) in enumerate(pbar):
            if contrast: 
                # choose a kernel
                kernel = KERNEL_DICT[kernel_type]
                conv = nn.Conv2d(3, 1, kernel_size=3, padding=1, groups=1, stride=1, bias=False)
                conv.weight.data = jt.Var([kernel, kernel, kernel]).float().unsqueeze(0)
            
                # get the image after being convolutioned by that kernel
                new_image = conv(images).detach().repeat(1,3,1,1)

                # append to input list
                images = jt.concat([images,new_image])
                output = model(images)
                total_acc += criterion(output)
                total_num += labels.shape[0]
            
            elif vae:
                output = model(images)
                batch_size = int(images.shape[0])
                output = jt.reshape(output, (batch_size,224,224))
                images = images.mean(dim=1)
                acc = criterion(output, images)
                total_acc += acc
                total_num += labels.shape[0]

            else:
                output = model(images)
                pred = np.argmax(output.data, axis=1)

                acc = np.sum(pred == labels.data)
                total_acc += acc
                total_num += labels.shape[0]

            pbar.set_description(f'Epoch {epoch} acc={total_acc / total_num:.2f}')

    acc = total_acc / total_num
    return acc


def test_one_epoch(model, test_loader):

    modelname = "VIT"
    idx2label = test_loader.classes
    model.eval()
    total_acc = 0
    total_num = 0

    total_labels = 102
    matrix = np.zeros((total_labels,total_labels)) # [实际][预测]
    plt.figure(figsize=(30, 30))

    pbar = tqdm(test_loader, desc=f'Testing')
    for i, (images, labels) in enumerate(pbar):
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]

        pbar.set_description(f'Test acc={total_acc / total_num:.2f}')
        
        _pred = idx2label[pred[0]]
        _label = idx2label[labels.data[0]]
        matrix[(int)(_label)][(int)(_pred)] += 1 # 混淆矩阵

    FP = matrix.sum(axis = 0) # 列  prec
    FN = matrix.sum(axis = 1) # recall

    for i in range(0,total_labels):
        TP = matrix[i][i]
        FP[i] = TP/(FP[i])
        FN[i] = TP/(FN[i])

    worst_fp = np.argmin(FP)
    worst_fn = np.argmin(FN)
    
    print("**********prec***********")
    print(worst_fp)
    print(FP[worst_fp])
    print(matrix[...,worst_fp].argsort()[::-1][0:3])
    print(matrix[...,worst_fp])
    print("**********recall***********")
    print(worst_fn)
    print(FN[worst_fn])
    print(matrix[worst_fn].argsort()[::-1][0:3])
    print(matrix[worst_fn])
    print("*********************")

    np.save("./log/matrix/"+modelname,matrix)
    ax = sns.heatmap(matrix,annot=True,square=True,cmap = 'YlGnBu')
    ax.set_xlabel("pred",fontsize=40)
    ax.set_ylabel("true",fontsize=40)
    ax.set_title (modelname,fontsize =50)
    fig = ax.get_figure()
    sns.plt.show()
    # fig.savefig("./output/heatmap_"+modelname+".png")
    fig.clf()


    ax = plt.subplots()
    plt.bar(np.arange(total_labels),FP)
    plt.xlabel("label")
    plt.title("prec "+modelname)
    # plt.savefig("./output/prec_"+modelname+".png")
    plt.show()
    plt.clf()

    ax = plt.subplots()
    plt.bar(np.arange(total_labels),FN)
    plt.xlabel("label")
    plt.title("recall "+modelname)
    # plt.savefig("./output/recall_"+modelname+".png")
    plt.show()
    plt.clf()

    acc = total_acc / total_num
    return acc
