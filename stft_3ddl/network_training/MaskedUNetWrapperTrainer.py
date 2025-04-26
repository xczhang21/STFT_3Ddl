import logging
import os
import sys
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.evaluation_method import compute_masked_metrics
from utilities.evaluation_method import log_masked_metrics

from utilities.target_mask_preds import save_TMMPs


def MaskedUNetWrapper_trainer_das1k(args, model, snapshot_path):
    from datasets.dataset_das1k_masked import masked_das1k_dataset
    from utilities.losses import MaskedMSELoss
    logging.basicConfig(
        filename=snapshot_path + '/log.txt',
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d %(message)s]',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    bd_train = masked_das1k_dataset(
        base_dir=args.dataset.root_path,
        list_dir=args.dataset.list_dir,
        split='train',
        mask_ratio=args.dataset.mask_ratio,
        mask_mode=args.dataset.mask_mode
    )
    db_val = masked_das1k_dataset(
        base_dir=args.dataset.root_path,
        list_dir=args.dataset.list_dir,
        split='test',
        mask_ratio=args.dataset.mask_ratio,
        mask_mode=args.dataset.mask_mode
    )

    writer = SummaryWriter(snapshot_path + '/log')

    print("The length of trian set is {}".format(len(bd_train)))

    def work_init_fn(worker_id):
        random.seed(args.seed +worker_id)

    trainloader = DataLoader(
        bd_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=work_init_fn
    )
    
    valloader = DataLoader(
        db_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion = MaskedMSELoss()
    # 下面的optimizer、scheduler与分类任务所选用的不同，不知对后续移植是否有影响
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        running_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        all_train_masks = []

        for step, sampled_batch in enumerate(trainloader):
            model.train()
            masked_input = sampled_batch['masked_input'].cuda()
            original = sampled_batch['original'].cuda()
            mask = sampled_batch['mask'].cuda()
            # 前向传播
            pred, used_mask = model(masked_input, mask)
            loss = criterion(pred, original, used_mask)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            running_loss += loss.item()

            all_train_preds.append(pred.detach())
            all_train_targets.append(original.detach())
            all_train_masks.append(used_mask.detach())
        
        # 更新学习率
        scheduler.step()

        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_targets = torch.cat(all_train_targets, dim=0)
        all_train_masks = torch.cat(all_train_masks, dim=0)

        avg_train_loss = running_loss / len(trainloader)
        train_metrics = compute_masked_metrics(all_train_preds, all_train_targets, all_train_masks)

        log_masked_metrics(writer, "train", avg_train_loss, train_metrics, epoch_num)
        
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            all_val_preds = []
            all_val_targets = []
            all_val_masks = []
            all_val_masked_targets = []
            all_TMMPs = []
            for val_data in db_val.datas:
                original = val_data['spectrum']
                id = val_data['id']
                _, original, masked_input, mask = db_val._get_item(id, original)

                original = original.unsqueeze(0).cuda()
                masked_input = masked_input.unsqueeze(0).cuda()
                mask = mask.unsqueeze(0).cuda()
                # masked_input = val_data['masked_input'].cuda()
                # original = val_data['original'].cuda()
                # mask = val_data['mask'].cuda()

                pred, used_mask = model(masked_input, mask)
                loss = criterion(pred, original, mask)

                running_val_loss += loss.item()

                all_TMMPs.append({
                        'id':val_data['id'],
                        'target':original.detach().cpu(),
                        'mask':used_mask.detach().cpu(),
                        'masked_target':masked_input.detach().cpu(),
                        'pred':pred.detach().cpu()
                        })
                all_val_preds.append(pred.detach())
                all_val_targets.append(original.detach())
                all_val_masks.append(used_mask.detach())

            all_val_preds = torch.cat(all_val_preds, dim=0)
            all_val_targets = torch.cat(all_val_targets, dim=0)
            all_val_masks = torch.cat(all_val_masks, dim=0)

            avg_val_loss = running_val_loss / len(db_val)
            val_metrics = compute_masked_metrics(all_val_preds, all_val_targets, all_val_masks)
            log_masked_metrics(writer, "val", avg_val_loss, val_metrics, epoch_num)        

            logging.info(f"epoch:{epoch_num} val_loss:{avg_val_loss:.5f} val_mse:{val_metrics['mse']:.5f} val_mae:{val_metrics['mae']:.5f} val_rmse:{val_metrics['rmse']:.5f}")
        
        # 保存targets, masks, masked_targets, preds
        save_TMMP_interval = args.save_CAM_interval # TMMP生成周期
        # save_TMMP_num = 0 # 对db_val的第几个sample进行TMMP的保存
        save_TMMP_num = None # 对db_val的所有的sample进行TMMP的保存
        save_TMMPs(epoch_num, all_TMMPs, writer, save_TMMP_interval, save_TMMP_num)

        # 最后一轮保存encoder
        if epoch_num == args.max_epochs -1:
            save_mode_path = os.path.join(snapshot_path, f"pretrained_encoder_epoch{epoch_num}.pth")
            torch.save(model.encoder.state_dict(), save_mode_path)
            logging.info(f"Save pretrained encoder to: {save_mode_path}")
    
    writer.close()
    return f"val_loss:{avg_val_loss:.5f}\t val_mse:{val_metrics['mse']:.5f}\t val_mae:{val_metrics['mae']:.5f}\t val_rmse:{val_metrics['rmse']:.5f}"


# 文件测试
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from networks_architecture.unet_class_configs import get_UNet_config
    from networks_architecture.masked_unet_wrapper import MaskedUNetWrapper
    import ml_collections

    # 构建配置
    config = ml_collections.ConfigDict()
    config.base_lr = 0.01
    config.batch_size = 32
    config.num_workers = 1
    config.seed = 1234
    config.max_epochs = 150
    config.save_TMMP_interval = 1

    config.dataset = ml_collections.ConfigDict()
    config.dataset.list_dir = "/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase"
    config.dataset.root_path = "/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/matrixs/scale_64"
    config.dataset.mask_ratio = 0.5
    config.dataset.mask_mode = "random"

    # snapshot_path 定义
    snapshot_path = f"../model/MaskedPretrain_das1k_masked_unet_epo{config.max_epochs}_bs{config.batch_size}_lr{config.base_lr}_mask{int(config.dataset.mask_ratio * 100)}"
    os.makedirs(snapshot_path, exist_ok=True)

    # 构建模型
    unet_config = get_UNet_config()
    model = MaskedUNetWrapper(unet_config).cuda()
    
    # 训练
    result = MaskedUNetWrapper_trainer_das1k(args=config, model=model, snapshot_path=snapshot_path)
    print(result)
    