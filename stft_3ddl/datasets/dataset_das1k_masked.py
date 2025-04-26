import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

MASK_MODEs = ['random', 'block', 'grid']

class masked_das1k_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split='train', mask_ratio=0.5, mask_mode='random'):
        self.split = split
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode

        if self.split == 'train':
            self.sample_list = open(os.path.join(list_dir, 'train' + '.txt')).readlines()
        elif self.split == 'test':
            self.sample_list = open(os.path.join(list_dir, 'test' + '.txt')).readlines()
        self.base_dir = base_dir
        self.datas = self._load_data()

    def _load_data(self):
        samples = []
        for line in self.sample_list:
            sample_id, _ = line.strip().split(' ')
            data_path = os.path.join(self.base_dir, sample_id, f"{sample_id}.npz")
            orig_data = np.load(data_path, allow_pickle=True)
            orig_data = orig_data[orig_data.files[0]]
            data = {
                'id': orig_data.tolist()['data_id'],
                'spectrum': orig_data.tolist()['spectrum'],
            }
            samples.append(data)
        return samples
    
    def _apply_mask(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        mask = torch.zeros_like(x)

        if self.mask_mode == 'random':
            num_mask = int(self.mask_ratio * x.numel())
            indices = random.sample(range(x.numel()), num_mask)
            flat_mask = mask.reshape(-1)
            flat_mask[indices] = 1
            mask = flat_mask.view_as(x)

        elif self.mask_mode == 'block':
            H, W = x.shape[-2], x.shape[-1]
            block_size = int((self.mask_ratio * H * W) ** 0.5)
            top = random.randint(0, H - block_size)
            left = random.randint(0, W - block_size)
            mask[top:top + block_size, left:left + block_size] = 1

        elif self.mask_mode == 'grid':
            H, W = x.shape[-2], x.shape[-1]
            stride = int((H * W / (self.mask_ratio * H * W)) ** 0.5)
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    mask[i, j] = 1

        else:
            raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

        x_masked = x.clone()
        x_masked[mask.bool()] = 0.0
        return x_masked, mask
    
    def __len__(self):
        return len(self.datas)

    def _get_item(self, id, original):
        masked_input, mask = self._apply_mask(original)
        original = torch.from_numpy(original.astype(np.float32)).unsqueeze(0)
        masked_input = masked_input.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return [id, original, masked_input, mask]
        
    
    def __getitem__(self, idx):
        original = self.datas[idx]['spectrum']
        id = self.datas[idx]['id']
        items =  self._get_item(id, original)
        return {
            'id': items[0],
            'original': items[1],
            'masked_input': items[2],
            'mask': items[3]
        }
        # masked_input, mask = self._apply_mask(original)
        # original = torch.from_numpy(original.astype(np.float32)).unsqueeze(0)
        # masked_input = masked_input.unsqueeze(0)
        # mask = mask.unsqueeze(0)

        # return {
        #     'id': self.datas[idx]['id'],
        #     'original': original,
        #     'masked_input': masked_input,
        #     'mask': mask
        # }


# 模块测试
if __name__ == '__main__':
    base_dir = '/home/zhang/zxc/STFT_3DDL/DATASETS/preprocessed_data/DAS1K/phase/matrixs/scale_256'
    list_dir = '/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/lists/DAS1K/phase'

    train_dataset = masked_das1k_dataset(base_dir=base_dir, list_dir=list_dir, split='train', mask_ratio=0.1, mask_mode='grid')
    print(f"Train set size: {len(train_dataset)}")

    sample = train_dataset[0]
    print("Sample ID:", sample['id'])
    print("Original shape:", sample['original'].shape)
    print("Masked input shape:", sample['masked_input'].shape)
    print("Mask shape:", sample['mask'].shape)

    # 可视化看看遮挡比例是否正确
    total_pixels = sample['mask'].numel()
    masked_pixels = sample['mask'].sum().item()
    print(f"Masked ratio: {masked_pixels / total_pixels:.2%}")

    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utilities.visualization import visualize_masked_sample

    sample = train_dataset[0]
    visualize_masked_sample(
    sample['original'], sample['masked_input'], sample['mask'],
    save_path=f"/home/zhang/zxc/STFT_3DDL/STFT_3Ddl/stft_3ddl/datasets/mask_vis/{sample['id']}.png"
)