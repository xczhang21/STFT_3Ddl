import logging
import sys
import random
from torchvision.transforms import transforms
from torch.utils.data import DataLoader



def ResNet_trainer_das1k_dataset(args, model, snapshot_path):
    from datasets.dataset_das1k import ds1k_dataset, RandomGenerator

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
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = ds1k_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split='train',
        transform=transform
    )
    print("The length of train set is: {}".format(len(db_train)))

    def work_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader