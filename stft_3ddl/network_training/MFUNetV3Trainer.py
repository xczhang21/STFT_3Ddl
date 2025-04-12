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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.grad_cam import GradCAM
from utilities.grad_cam import visualize_cam_multi
from utilities.grad_cam import save_grad_cam_multi

from utilities.confusion_matrix import  save_confusion_matrix

from utilities.pred_details import save_pred_details

from utilities.evaluation_method import compute_metrics
from utilities.evaluation_method import compute_map
from utilities.evaluation_method import log_metrics

def MFUNetV3_trainer_mfdas1k(args, model, snapshot_path):
    from datasets.dataset_mfdas1k import mfdas1k_dataset, RandomGenerator

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
    db_train = mfdas1k_dataset(
        base_dir=args.dataset.root_path,
        spectrum_size=args.dataset.spectrum_size,
        list_dir=args.dataset.list_dir,
        split='train',
        transform=transform
    )
    db_val = mfdas1k_dataset(
        base_dir=args.dataset.root_path,
        spectrum_size=args.dataset.spectrum_size,
        list_dir=args.dataset.list_dir,
        split='test',
        transform=transform
    )
    class_names = args.dataset.class_names
    
    writer = SummaryWriter(snapshot_path + '/log')
    
    print("The length of train set is: {}".format(len(db_train)))

    def work_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=work_init_fn,
    )
    mse_loss = MSELoss() # MSE适用于回归任务
    ce_loss = CrossEntropyLoss() # 交叉熵适用于分类任务

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.05)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        all_train_probs = []

        for step, sampled_batch in enumerate(trainloader):
            model.train()
            spectrums_batch, label_batch = sampled_batch['spectrums'], sampled_batch['label']
            spectrums_batch = [spectrum_batch.cuda() for spectrum_batch in spectrums_batch]
            label_batch = label_batch.long().cuda()
            # 前向传播
            outputs = model(spectrums_batch[1], spectrums_batch[0])
            loss = ce_loss(outputs, label_batch)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1) # 转换为概率分布
            predictions = torch.argmax(probs, dim=1)

            all_train_labels.append(label_batch)
            all_train_preds.append(predictions)
            all_train_probs.append(probs)

    
        # 更新学习率
        scheduler.step()

        all_train_labels = torch.cat(all_train_labels)
        all_train_preds = torch.cat(all_train_preds)
        all_train_probs = torch.cat(all_train_probs)

        avg_train_loss = running_loss / len(trainloader)
        train_metrics = compute_metrics(all_train_labels, all_train_preds, num_classes)
        train_map = compute_map(torch.nn.functional.one_hot(all_train_labels, num_classes), all_train_probs, num_classes)

        log_metrics(writer, "train", avg_train_loss, train_metrics, train_map, epoch_num)


        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            all_val_spectrums = []
            all_val_outputs = []
            all_val_labels = []
            all_val_preds = []
            all_val_probs = []
            for val_data in db_val.datas:
                spectrums = RandomGenerator.divisive_crop(val_data['spectrums'])
                spectrum_id = val_data['id']
                all_val_spectrums.append({'spectrums':spectrums, 'id':spectrum_id})
                label = RandomGenerator.label_convert(val_data['label']).unsqueeze(0).cuda()

                spectrums = [spectrum.unsqueeze(0).cuda() for spectrum in spectrums]
                output = model(spectrums[1], spectrums[0])
                probs = torch.softmax(output, dim=1)

                loss = ce_loss(output, label.long())
                running_val_loss += loss.item()

                all_val_outputs.append(output)
                all_val_labels.append(label)
                all_val_preds.append(torch.argmax(probs, dim=1))
                all_val_probs.append(probs)
            
            all_val_outputs = torch.cat(all_val_outputs)
            all_val_labels = torch.cat(all_val_labels)
            all_val_preds = torch.cat(all_val_preds)
            all_val_probs = torch.cat(all_val_probs)

            val_metrics = compute_metrics(all_val_labels, all_val_preds, num_classes)
            val_map = compute_map(torch.nn.functional.one_hot(all_val_labels.long(), num_classes), all_val_probs, num_classes)

            avg_val_loss = running_val_loss / len(db_train.datas)

            log_metrics(writer, "val", avg_val_loss, val_metrics, val_map, epoch_num)

            logging.info(f"epoch:{epoch_num} val_loss:{avg_val_loss:.5f} val_acc:{val_metrics['accuracy']:.5f} val_pre:{val_metrics['precision']:.5f}")

        # 保存混淆矩阵
        if epoch_num == max_epoch-1:
            save_confusion_matrix(writer, all_val_labels, all_val_outputs, num_classes, class_names)
            
        # 保存CAM
        save_CAM_interval = args.save_CAM_interval # grad_CAM生成周期
        # save_CAM_num = 0 # 对val_spectrums的第几张进行grad_CAM
        save_CAM_num = None # 对val_spectrums的多有特征进行grad_CAM
        # save_grad_cam(epoch_num, val_spectrums, model, model.decoder1, writer, save_CAM_interval, save_CAM_num)
        # 因为MFUNet是多输入结构，所以修改save_grad_cam为save_grad_cam_multi以适应MFUNet
        save_grad_cam_multi(
            epoch_num,
            all_val_spectrums,
            model,
            [model.fuse4, model.fuse4, model.decoder1],
            writer,
            save_CAM_interval,
            save_CAM_num
        )

        
        # 保存预测详情
        if epoch_num == max_epoch-1:
        # if epoch_num == 1:
            save_pred_details(writer, all_val_labels, all_val_outputs, num_classes, class_names, db_val.sample_list, epoch_num)
        
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return f"val_loss:{avg_val_loss:.5f}\t val_acc:{val_metrics['accuracy']:.5f}\t val_pre:{val_metrics['precision']}"


# 文件测试
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from networks_architecture.mulit_feature_unet_v3_class_configs import get_MFUNetV3_config
    from networks_architecture.mulit_feature_unet_v3_class_modeling import MFUNetV3
    import ml_collections
    import argparse

    config = ml_collections.ConfigDict()

    config.base_lr = 0.01
    config.batch_size = 16
    config.dataset = ml_collections.ConfigDict()
    config.dataset.list_dir = "/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase"
    config.dataset.num_channels = 1
    config.dataset.num_classes = 10
    config.dataset.root_path = "/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K"
    config.dataset.class_names = ['cathorn', 'drilling', 'footsteps', 'handhammer', 'handsaw', 'jackhammer',
                          'rain', 'shoveling', 'thunderstorm', 'welding']
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
    config.net_name = "MFUNetV3"
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

    # 控制数据加载的线程数
    config.num_workers = 1

    # 控制CAM保存间隔
    config.save_CAM_interval = 1

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_net = get_MFUNetV3_config()
    network = MFUNetV3(config=config_net)
    network = network.cuda()

    trainer = MFUNetV3_trainer_mfdas1k(args=config, model=network, snapshot_path=snapshot_path)