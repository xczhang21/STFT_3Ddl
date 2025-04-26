import matplotlib.pyplot as plt
import torch

def visualize_masked_sample(original, masked_input, mask, title=None, save_path=None):
    """
    original: torch.Tensor or numpy.ndarray, shape: [H, W] or [1, H, W]
    masked_input: same shape as original
    mask: same shape as original
    """
    # 确保都是 [H, W]
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu()
        if original.ndim == 3:
            original = original.squeeze(0)
        original = original.numpy()
    if isinstance(masked_input, torch.Tensor):
        masked_input = masked_input.detach().cpu()
        if masked_input.ndim == 3:
            masked_input = masked_input.squeeze(0)
        masked_input = masked_input.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = mask.numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original, cmap='viridis')
    axs[0].set_title("Original")
    axs[1].imshow(masked_input, cmap='viridis')
    axs[1].set_title("Masked Input")
    axs[2].imshow(mask, cmap='gray')
    axs[2].set_title("Mask")

    for ax in axs:
        ax.axis('off')
    if title:
        fig.suptitle(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close()
