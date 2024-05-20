# coding=gb2312
import os
import torchvision
from torch.utils.data import DataLoader
import dataset
import copy
import time
import torch
import matplotlib.pyplot as plt
from model_utils import resnext34,resnext50  # import的模型可根据实际进行修改
import torch.nn as nn
import pandas as pd


#xye,↓以下自己训练代码没用其他库↓


def xyeData(batch_size=32,num_workers=0,blur_p=None,ers_p=None):#xye,dataloder
    # data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(90, fill=(255, 255, 255)),#旋转增强
            torchvision.transforms.Resize((256, 224)),
            torchvision.transforms.CenterCrop((256, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 224)),
            torchvision.transforms.CenterCrop((256, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    if blur_p:
        data_transforms['train']=torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(90, fill=(255, 255, 255)),#旋转增强
            torchvision.transforms.Resize((256, 224)),
            torchvision.transforms.CenterCrop((256, 224)),
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(3, 3), sigma=1)], p=blur_p),#xye,模糊增强
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if ers_p:
        data_transforms['train']=torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(90, fill=(255, 255, 255)),#旋转增强
            torchvision.transforms.Resize((256, 224)),
            torchvision.transforms.CenterCrop((256, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            torchvision.transforms.RandomErasing(p=ers_p, scale=(0.01, 0.03)),#xye,随机擦除增强(p=0.5, scale=(0.02,0.4), ratio=(0.3,3), value=(255,255,255)
        ])
    image_datasets = {}

    prepath=os.path.dirname(os.path.dirname(__file__))+'/data/'
    image_datasets['train'] = dataset.SingleTaskDataset(prepath+'data2/train.csv', 'Task_Chinese_medicinal_herb', ',', prepath, data_transforms['train'])
    image_datasets['val'] = dataset.SingleTaskDataset(prepath+'data2/test.csv', 'Task_Chinese_medicinal_herb', ',', prepath, data_transforms['val'])

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes

def train_model(start_epoch, batch_size, print_freq, save_path, save_epoch_freq, model, criterion, optimizer, scheduler, num_epochs, dataset_sizes):
    since = time.time()
    resumed = False
    best_val_acc=0#xyeLOG
    best_val_epoch=0#xyeLOG
    this_train_acc=0#xyeLOG
    best_train_acc=0#xyeLOG
    if resume:# xyeLOG,resume
        with open(os.path.join(save_path, "log.txt"), "r") as file:
            best_val_epoch = file.read()
            if"【"in best_val_epoch:
                best_val_epoch=best_val_epoch.split("【")
                best_val_acc=float(best_val_epoch[-3].split("】")[0])
                best_train_acc=float(best_val_epoch[-2].split("】")[0])
                this_train_acc=float(best_val_epoch[-2].split("】")[0])
                best_val_epoch=int(best_val_epoch[-1].split("】")[0])

    best_model_wts = model.state_dict()

    for epoch in range(start_epoch+1,num_epochs):

        # 同时进行训练和测试,测试不会BP
        for phase in ['train','val']:
            if phase == 'train':
                if scheduler:
                    if start_epoch > 0 and (not resumed):
                        scheduler.step(start_epoch+1)
                        resumed = True
                    else:
                        scheduler.step(epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloders[phase]):
                # wrap them in Variable
                if use_gpu:
                    inputs = torch.autograd.Variable(inputs.cuda())
                    labels = torch.autograd.Variable(labels.cuda())
                else:
                    inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                smooth_strength = 0 # xye,平滑标签
                if smooth_strength==0:
                    loss = criterion(outputs, labels)
                else:
                    target_one_hot = torch.zeros_like(outputs)
                    target_one_hot[torch.arange(len(labels)), labels] = 1
                    target_one_hot = (target_one_hot[:] * (1.0 - smooth_strength)) + (torch.tensor([smooth_strength]).to(outputs.device)/outputs.shape[1])
                    loss = criterion(outputs, target_one_hot)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data#xyeERR,default: += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / ((i+1)*batch_size)
                batch_acc = running_corrects / ((i+1)*batch_size)

                if phase == 'train' and i%print_freq == 0:
                    if scheduler:
                        now_lr=scheduler.get_lr()[0]
                    else:
                        now_lr=optimizer.param_groups[0]['lr']
                    print('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                          epoch, num_epochs - 1, i, round(dataset_sizes[phase]/batch_size)-1, now_lr,
                        phase, batch_loss, batch_acc, print_freq/(time.time()-tic_batch)))
                    tic_batch = time.time()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if not os.path.exists(save_path):#xyeLOG
                os.makedirs(save_path)
            with open(os.path.join(save_path, "log.txt"), "a") as file:#xyeLOG
                file.write(
                    'ep{} {} Loss: {:.4f} Acc: {:.4f}\n'.format(epoch, phase, epoch_loss, epoch_acc)
                )
            if phase=='train':
                this_train_acc=epoch_acc
            elif (epoch_acc>best_val_acc)or(epoch_acc==best_val_acc and this_train_acc>best_train_acc):#xyeLOG
                best_val_acc=epoch_acc
                best_val_epoch=epoch
                best_train_acc=this_train_acc


        if (epoch+1) % save_epoch_freq == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # torch.save(model, os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth.tar"))
            torch.save(model.module.state_dict() , os.path.join(save_path, "epoch_" + str(epoch) + ".pth"))#xyeDICTsave

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    with open(os.path.join(save_path, "log.txt"), "a") as file:  #xyeLOG
        file.write(
            'best_val_acc【{:.4f}】 with train_acc【{:.4f}】 in best_val_epoch【{}】\n'.format(best_val_acc, best_train_acc, best_val_epoch)
        )

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    """参数调节"""
    num_workers=0
    print_freq=10
    save_epoch_freq=1
    num_class=14
    num_epochs=200
    gpus="0,1"
    lr=1e-3#hp
    batch_size=32#hp
    op="adam"
    blur_p=None
    ers_p=None

    save_path = "output"
    ex=os.path.basename(__file__).split(".")[1]
    if ex=="py":gpus="0"
    save_path="output"+ex
    resume=""#save_path+"/epoch_180.pth"
    print(ex)
    if ex[2:4]=="13":lr=1e-3
    elif ex[2:4]=="53":lr=5e-3
    elif ex[2:4]=="54":lr=5e-4
    elif ex[2:4]=="14":lr=1e-4
    else:exit("错误")
    if ex[5]=="b":
        blur_p=float(ex[6])/10
    elif ex[5]=="e":
        ers_p=float(ex[6])/10
    print(blur_p)
    print(ers_p)

    model = resnext34(num_classes=num_class)
    # model = resnext50(num_classes=num_class, groups=32, width_per_group=4)
    # #xye:beta test,with downloaded pretrain weight
    # model = torchvision.models.resnext50_32x4d(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, num_class)

    # 读数据
    dataloders, dataset_sizes = xyeData(batch_size,num_workers,blur_p,ers_p)

    # 掉电恢复
    start_epoch=0
    if resume:
        if os.path.isfile(resume):
            print(("=> loading checkpoint '{}'".format(resume)))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint)#xye
            start_epoch=int(resume.split("_")[-1].split(".")[0])#xye
            print("[] continue start in epoch"+str(start_epoch))#xye
        else:
            print(("=> no checkpoint found at '{}'".format(resume)))

    # 使用显卡训练,默认0卡1卡训练
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))
    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in gpus.strip().split(',')])

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    if op=="sgd":
        optimizer_ft = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)  # SGD学习率调度器, Decay LR by a factor of 0.1 every 7 epochs
    elif op=="adam":
        optimizer_ft = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        exp_lr_scheduler = None
    else:exit("优化器指定错误")

    model = train_model(start_epoch, batch_size, print_freq, save_path, save_epoch_freq,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=num_epochs,
                           dataset_sizes=dataset_sizes)