import logging
import sys
import random
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import os
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.grad_cam import GradCAM
from utilities.grad_cam import visualize_cam


def MFUNet_trainer_das1k(args, model, snapshot_path):
    from datasets.dataset_mfdas1k import mfds1k_dataset, RandomGenerator

    transform = transforms.Compose([
        RandomGenerator()
    ])
    logging.basicConfig(
        filename=snapshot_path + '/log.txt',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d %(message)s]',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.dataset.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = mfds1k_dataset(
        base_dir=args.dataset.root_path,
        pectrum_size=args.dataset.spectrum_size,
        list_dir=args.dataset.list_dir,
        split='train',
        transform=transform
    )
    db_val = mfds1k_dataset(
        base_dir=args.dataset.root_path,
        pectrum_size=args.dataset.spectrum_size,
        list_dir=args.dataset.list_dir,
        split='test',
        transform=transform
    )
    print("The length of train set is: {}".format(len(db_train)))

    def work_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        worker_init_fn=work_init_fn,
    )
    mse_loss = MSELoss() # MSE适用于回归任务
    ce_loss = CrossEntropyLoss() # 交叉熵适用于分类任务

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.05)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    writer = SummaryWriter(snapshot_path + '/log')
    tier_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        running_loss = 0.0
        train_correct = 0
        val_correct = 0
        total = 0
        val_total = 0
        for step, sampled_batch in enumerate(trainloader):
            model.train()
            spectrums_batch, label_batch = sampled_batch['spectrums'], sampled_batch['label']
            spectrums_batch = [spectrum_batch.cuda() for spectrum_batch in spectrums_batch]
            label_batch = label_batch.long().cuda()
            # 前向传播
            outputs = model(spectrums_batch[0], spectrums_batch[1])
            loss = ce_loss(outputs, label_batch)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            # 计算精度
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == label_batch).sum().item()
            total += label_batch.size(0)

        # 更新学习率
        scheduler.step()

        # 计算平均损失和精度
        avg_loss = running_loss / len(trainloader)
        accuracy = train_correct / total
        writer.add_scalar('Loss/train', avg_loss, epoch_num)
        writer.add_scalar('Accuracy/train', accuracy, epoch_num)

        model.eval()
        with torch.no_grad():
            val_spectrums = []
            val_labels = torch.Tensor().cuda()
            val_outputs = torch.Tensor().cuda()
            for val_data in db_val.datas:
                spectrums = RandomGenerator.divisive_crop(val_data['spectrums'])
                spectrum_id = val_data['id']
                val_spectrums.append({'spectrums':spectrums, 'id':spectrum_id})
                spectrums = [spectrum.unsqueeze(0) for spectrum in spectrums]
                label = RandomGenerator.label_convert(val_data['label'])
                label = label.unsqueeze(0)
                spectrums = [spectrum.cuda() for spectrum in spectrums]
                label = label.cuda()
                output = model(spectrums[0], spectrums[1])
                val_labels = torch.cat((val_labels, label))
                val_outputs = torch.cat((val_outputs, output))

            # 计算损失
            val_loss = ce_loss(val_outputs, val_labels.long()).cpu().detach().numpy()
            
            # 计算精度
            predictions = torch.argmax(val_outputs, dim=1)
            val_correct += (predictions == val_labels).sum().item()
            val_total += val_labels.size(0)

            # 计算平均精度
            accuracy = val_correct / val_total
            logging.info(f'epoch:{epoch_num} val_loss:{val_loss:.5f} val_acc:{accuracy:.5f}')
            writer.add_scalar('Loss/val', val_loss, epoch_num)
            writer.add_scalar('Accuracy/val', accuracy, epoch_num)
        
        save_CAM_interval = 5 # grad_CAM生成周期
        save_CAM_num = 0 # 对val_spectrums的第几张进行grad_CAM
        if epoch_num == 0:
            spectrum_id = val_spectrums[save_CAM_num]['id']
            spectrum = val_spectrums[save_CAM_num]['spectrums'][0]
            writer.add_image(f'{spectrum_id}', torch.flip(spectrum, dims=[1]), 0, dataformats='CHW')

        if (epoch_num + 1) % save_CAM_interval == 0:
            # 每20次迭代，保存一次spectrum 图和 grad_CAM图
            spectrum_id = val_spectrums[save_CAM_num]['id']
            spectrums = val_spectrums[save_CAM_num]['spectrums']

            grad_cam = GradCAM(model, target_layer=model.decoder1)
            # grad_cam = GradCAM(model, target_layer=model.layer4[-1].conv3)
            # grad_cam = GradCAM(model, target_layer=model.layer3)
            
            cam = grad_cam.generate_cam([spectrum.unsqueeze(0).cuda() for spectrum in spectrums], target_class=None, target_batch=0)
            if spectrum.shape[0] == 0:
                superimposed_image = visualize_cam(cam, spectrum)
            else:
                superimposed_image = visualize_cam(cam, spectrum[:1])
            writer.add_image(f'{spectrum_id}/epoch_{epoch_num}', superimposed_image, 1, dataformats='HWC')
        
        # save_interval = 50
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return f"val_loss:{val_loss:.5f}\t val_acc:{accuracy:.5f}"





# 文件测试
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from networks_architecture.mulit_feature_unet_class_configs import get_MFUNet_config
    from networks_architecture.mulit_feature_unet_class_modeling import MFUNet
    import ml_collections
    import argparse

    config = ml_collections.ConfigDict()

    config.base_lr = 0.01
    config.batch_size = 32
    config.dataset = ml_collections.ConfigDict()
    config.dataset.list_dir = "/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase"
    config.dataset.num_channels = 1
    config.dataset.num_classes = 10
    config.dataset.root_path = "/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K"
    config.dataset.spectrum_size = 256
    config.dataset_name = "mfdas1k"
    config.dataset_spectrum_size = 256
    config.is_pretrain = False
    config.max_epochs = 150
    config.n_gpu = 1
    config.net = ml_collections.ConfigDict()
    # config.net.block = "Bottleneck"
    # config.net.encoder_num = [3, 4, 24, 3]
    # config.net.groups = None
    config.net.in_channels = 1
    # config.net.norm_layer = None
    config.net.num_classes = 10
    # config.net.replace_stride_with_dilation = None
    # config.net.width_per_group = None
    # config.net.zero_init_residual = None
    config.net_name = "MFUNet"
    config.seed = 1234
    config.task_type = "cla"


    assert config.task_type == 'cla', f"task_type({config.task_type}) is not cla"
    dataset_name = config.dataset_name
    exp = 'STFT_3Ddl_' + config.dataset_name + '_test'

    snapshot_path = "../model/{}/".format(exp)
    snapshot_path = snapshot_path + 'traniner_test'
    snapshot_path = snapshot_path + '_' + config.net_name
    snapshot_path = snapshot_path + '_epo' + str(config.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(config.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(config.base_lr)
    snapshot_path = snapshot_path + '_ssize' + str(config.dataset_spectrum_size)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_net = get_MFUNet_config()
    network = MFUNet(config=config_net)
    network = network.cuda()

    trainer = MFUNet_trainer_das1k(args=config, model=network, snapshot_path=snapshot_path)