# coding=gb2312
import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import metrics
from tqdm import tqdm
import dataset
from model_utils import resnext34  # import的模型可根据实际修改
# from model_utils import resnet18  # import的模型可根据实际修改
# from model_utils import GoogLeNet
'''
    test.py可修改的部分仅四部分：
        1、from model_utils import resnet18  # import的模型可根据实际修改
        2、parser.add_argument命令行参数，只可修改 default 字段
        3、main函数部分的模型调用：model = resnet18(num_classes=total_classes, include_top=True)  # 可根据实际调用模型
        4、数据加载部分，image_datasets和test_dataset
'''


parser = argparse.ArgumentParser(description='Classification Competition Project')
parser.add_argument('--name', default='赛博神农', type=str, help='name of the run')
parser.add_argument('--seed', default=2, type=int, help='random seed')
parser.add_argument('--test',
                    # default=os.path.dirname(os.path.dirname(__file__))+'/data/data2/test.csv',#xye
                    default=os.path.dirname(os.path.dirname(__file__))+'/data/test.csv',#xye
                    type=str, help='path to test image files/labels')
parser.add_argument('--sep', default=',', type=str, help='column separator used in csv(default: ",")')
parser.add_argument('--data-dir', default=os.path.dirname(os.path.dirname(__file__))+'/data/',
                    type=str, help='root directory of images')  # images存放目录(建议根目录)，注意参考csv中的image_path，否则读取不成功
parser.add_argument('--best-state-path',
                    default=os.path.dirname(__file__)+'/best_state=epoch_250.pth',#xye
                    type=str,
                    help='path to load best state')
parser.add_argument('--batch-size', default=64, type=int, help='batch size (default: 32)')
parser.add_argument('--task-name', default='Task_Chinese_medicinal_herb', type=str,
                    help='names of the task separated y comma')
parser.add_argument('--task-names', default='tongue_colour,moss_colour,tongue_shape,moss_quality', type=str,
                    help='names of the task separated y comma')

num_classes_tasks = {
    'tongue_colour': 5,
    'moss_colour': 3,
    'tongue_shape': 6,
    'moss_quality': 9,
    'Task_Chinese_medicinal_herb': 14
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def Multitask_test_model_save_results(model, dataloader, num_classes, class_names=None, device=None):
    print("Testing the model and saving the results:", flush=True)
    model.eval()

    num_tasks = len(num_classes)
    preds_ = [[] for _ in range(num_tasks)]
    labels_ = [[] for _ in range(num_tasks)]
    with torch.no_grad():
        for _, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test Iteration"):
            inputs = inputs.to(device)
            labels = [labels[i].to(device) for i in range(len(labels))]
            outputs = model(inputs)
            idx_start = 0
            for idx in range(num_tasks):
                output_idx = outputs[:, idx_start:idx_start + num_classes[idx]]
                label_idx = labels[idx]
                valid_idx = torch.nonzero(1 * (label_idx != -1)).view(-1)
                if len(valid_idx) == 0:
                    continue
                idx_start += num_classes[idx]
                _, preds = torch.max(output_idx[valid_idx], 1)
                preds_[idx].extend(preds.cpu().numpy().tolist())
                labels_[idx].extend(label_idx[valid_idx].data.cpu().numpy().tolist())

    f1_scores = []
    for idx in range(num_tasks):
        label_y = []
        label_pred = []
        task_class_names = class_names[idx]
        for i in range(len(preds_[idx])):
            label_y.append(task_class_names[labels_[idx][i]])
            label_pred.append(task_class_names[preds_[idx][i]])
        f1_score = metrics.f1_score(label_y, label_pred, average="macro")
        f1_scores.append(f1_score)
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    file_name = f"{parent_dir}/out/multi_output.txt"
    sum_f1_score = sum(f1_scores)
    average_f1_score = sum_f1_score/num_tasks
    print("f1值为:", average_f1_score)
    with open(file_name, "a") as file:
        file.write(str(average_f1_score)+"\n")
    print("测试结果已保存到", file_name)


def Singletask_test_model_save_results(model, dataloader, device=None):
    print("Testing the model and saving the results:", flush=True)

    model.eval()
    probs_ = []
    preds_ = []
    labels_ = []

    running_corrects = 0
    count = 0
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Test Iteration"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, 1)
            probs, preds = torch.max(outputs, 1)
            preds_.extend(preds.cpu().numpy().tolist())
            probs_.extend(probs.cpu().numpy().tolist())
            labels_.extend(labels.data.cpu().numpy().tolist())
            running_corrects += torch.sum(preds == labels.data)
            count += len(labels.data)
    f1_score = metrics.f1_score(labels_, preds_, average="macro")
    print(f1_score)
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    file_name = f"{parent_dir}/out/single_output.txt"
    with open(file_name, "a") as file:
        file.write(str(f1_score)+"\n")
    print("测试结果已保存到", file_name)


def main():
    global device
    use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(use_gpu)
    print("Using device: {}".format(use_gpu), flush=True)
    args = parser.parse_args()
    print(args, flush=True)
    set_seed(args.seed)
    sep = args.sep
    data_dir = args.data_dir
    best_state_path = args.best_state_path
    batch_size = args.batch_size
    task_name = args.task_name
    task_names = args.task_names.split(',')
    num_classes = [num_classes_tasks[x] for x in task_names]
    total_classes = sum(num_classes)
    model = resnext34(num_classes=14)  # 可根据实际调用模型xye
    # model = resnet18(num_classes=total_classes, include_top=True)  # 可根据实际调用模型
    # model = resnet18(num_classes=total_classes, include_top=True)  # 可根据实际调用模型
    # model = GoogLeNet(num_classes=14, aux_logits=True, init_weights=True)
    best_state = torch.load(best_state_path)
    model.load_state_dict(best_state)

    data_transforms = {
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 224)),
            torchvision.transforms.CenterCrop((256, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("Initializing Datasets and Dataloaders...", flush=True)
    image_datasets = {
                    'test': dataset.SingleTaskDataset(args.test, task_name, sep, data_dir, data_transforms['test'])}
                    # 'test': dataset.MultitaskDataset(args.test, sep, data_dir, data_transforms['test'], task_names)}

    class_names = image_datasets['test'].classes
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()), flush=True)
        model = torch.nn.DataParallel(model)

    if args.test:
        test_dataset = dataset.SingleTaskDataset(args.test, task_name, sep, data_dir, data_transforms['test'])
        # test_dataset = dataset.MultitaskDataset(args.test, sep, data_dir, data_transforms['test'], task_names)
        print("There are {} test images.".format(len(test_dataset)))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        # Multitask_test_model_save_results(model, test_dataloader, num_classes, class_names = class_names,
        #                                   device = device)
        Singletask_test_model_save_results(model, test_dataloader, device=device)


if __name__ == "__main__":
    main()