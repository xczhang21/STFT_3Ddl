# 把配置文件放在train_config.py中，不需要用太多的parer参数，只需要配置写入train_config.py，在train.py中只用get_test_train()获取这些参数
import argparse
import torch.backends.cudnn as cudnn
import random
import numpy as np
import torch
import os
from os.path import join as join
import sys

import train_configs as train_configs
import datasets.datasets_config as datasets_config
from utilities.utils import recursive_find_python_class
from utilities.utils import get_next_time_number


parser = argparse.ArgumentParser()
parser.add_argument('--train_config', type=str,
                    default='test_train', help='train配置文件在train_config.py中')
parser.add_argument('--prepro_method', type=str,
                    default=None, help='数据预处理方法，默认为空')
parser.add_argument('--times', type=str,
                    default=None, help="trainin times")
parser.add_argument('--deterministic', type=int, default=0,
                    help='whether use deterministic training')
args = parser.parse_args()


if __name__ == '__main__':

    if not args.deterministic:
        # 非确定性模式
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        # 确定性模式
        cudnn.benchmark = False
        cudnn.deterministic = True

    # 判断train_config是否存在且可调用
    if args.prepro_method == None:
        train_config_name = f"{args.train_config}"
    else:
        train_config_name = f"{args.prepro_method}_{args.train_config}"
    assert hasattr(train_configs, f"get_{train_config_name}"), f"Config 'get_{train_config_name}' does not exist in the train_config."
    assert callable(getattr(train_configs, f"get_{train_config_name}")), f"get_{train_config_name} is not callable."

    train_config = getattr(train_configs, f"get_{train_config_name}")()

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed(train_config.seed)

    # 统一模型的输入通道、分类数与数据集的通道数和分类数
    train_config.net.in_channels = train_config.dataset.num_channels
    train_config.net.num_classes = train_config.dataset.num_classes

    assert train_config.task_type == 'cla', f"Task_type {train_config.task_type} is not 'cla'"


    exp = 'STFT_3Ddl_' + train_config.dataset_name
    
    snapshot_path = "../model/{}/{}/".format(exp, train_config_name)
    snapshot_path = snapshot_path + train_config.net_name
    snapshot_path = snapshot_path + '_pretrain' if train_config.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(train_config.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(train_config.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(train_config.base_lr)
    # 先尝试获取 dataset_spectrum_size，如果不存在则尝试 dataset_scale_size
    attr_name = 'dataset_spectrum_size' if hasattr(train_config, 'dataset_spectrum_size') else 'dataset_scale_size'
    # 获取属性值
    attr_value = getattr(train_config, attr_name, None)
    # 拼接字符串
    snapshot_path = snapshot_path + '_ssize' + str(attr_value)

    snapshot_path_parent = snapshot_path

    if args.times == None:
        args.times = str(get_next_time_number(snapshot_path_parent))
    
    train_config.times = args.times
    
    snapshot_path = snapshot_path + "/{}/".format(args.times)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        # print(snapshot_path)

    network_name = train_config.net_name
    network_class = recursive_find_python_class([join(sys.path[0], "networks_architecture")],
                                                network_name, current_module="networks_architecture")
    assert network_class != None, f"Network {network_name} is not derived from networks_architecture"

    
    trainer_name = f"{getattr(train_config, 'trainer_prefix', '')}{train_config.net_name}_trainer_{train_config.dataset_name}"
    trainer_func = recursive_find_python_class([join(sys.path[0], "network_training")],
                                               trainer_name, current_module="network_training")
    assert trainer_func != None, f"Trainer {trainer_name} is not derived from network_training"

    network = network_class(config=train_config.net)
    network.cuda()

    trainer = trainer_func(train_config, model=network, snapshot_path=snapshot_path)
    with open("{}/results.txt".format(snapshot_path_parent), 'a') as file:
        file.write(f"times:{args.times}\t {trainer}\n")
    