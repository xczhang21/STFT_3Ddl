# -*- coding: utf-8 -*-
"""
Swin (timm) 封装：类名 Swin_timm
与你的 ResNet(config) 风格一致：传入 ml_collections.ConfigDict 的配置

期望的 config 字段：
    model_name: str           # timm 模型名，如 'swin_tiny_patch4_window7_224'
    in_channels: int          # 输入通道数（1/2/3）
    num_classes: int          # 类别数
    img_size: int             # 输入尺寸，常见 224 或 256
    pretrained: bool          # 是否加载 ImageNet 预训练
    drop_rate: float          # dropout，默认 0.0
    drop_path_rate: float     # drop-path，默认 0.1
    repeat_to_3ch: bool       # 当预训练要求 3 通道而你用 1/2 通道时复制到 3 通道
    enable_feat_hook: bool    # 是否注册最后阶段特征的 forward hook（Grad-CAM 使用）
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from timm import create_model
except Exception as e:
    raise ImportError(
        "缺少依赖：timm。请先安装：\n  pip install timm>=0.9.0\n原始错误：{}".format(e)
    )


class Swin_timm(nn.Module):
    """timm.Swin 的薄封装，提供标准 forward + 可选中间特征抓取"""

    def __init__(self, config):
        super().__init__()
        self.cfg = config

        # ------------ 通道处理开关 ------------
        self._feat_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.last_feat = None
        self._repeat_to_3 = bool(getattr(config, "repeat_to_3ch", False)) and int(
            config.in_channels
        ) in (1, 2)

        in_chans = 3 if self._repeat_to_3 else int(config.in_channels)
        num_classes = int(config.num_classes)
        img_size = int(getattr(config, "img_size", 224))
        drop_rate = float(getattr(config, "drop_rate", 0.0))
        drop_path_rate = float(getattr(config, "drop_path_rate", 0.0))
        pretrained = bool(getattr(config, "pretrained", False))
        model_name = str(getattr(config, "model_name", "swin_tiny_patch4_window7_224"))

        # ------------ 构建 timm backbone ------------
        # timm 大多数 Swin 变体支持 in_chans / img_size / num_classes 这些参数
        self.backbone = create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            img_size=img_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # ------------ 可选：注册最后阶段特征 hook（给 Grad-CAM 用） ------------
        if bool(getattr(config, "enable_feat_hook", True)):
            self._register_last_stage_hook()

    # ---------------- hooks ----------------
    def _register_last_stage_hook(self) -> None:
        """尝试在 backbone 末端注册 forward hook，不同 timm 版本命名略有差异"""
        candidates = []

        # 常见：layers[-1].blocks[-1].norm（Swin v1/v2）
        try:
            candidates.append(self.backbone.layers[-1].blocks[-1].norm)
        except Exception:
            pass

        # 备选：整体的 norm（head 之前）
        try:
            if hasattr(self.backbone, "norm"):
                candidates.append(self.backbone.norm)
        except Exception:
            pass

        def _hook(_m, _x, y):
            # y 可能是 (B, N, C) 或 (B, C, H, W)，保留即可
            self.last_feat = y

        for t in candidates:
            if t is None:
                continue
            try:
                self._feat_hook_handle = t.register_forward_hook(_hook)
                break
            except Exception:
                continue

    def remove_hooks(self) -> None:
        """显式释放 forward hook，避免长时间训练中的资源泄漏"""
        if getattr(self, "_feat_hook_handle", None) is not None:
            try:
                self._feat_hook_handle.remove()
            except Exception:
                pass
            finally:
                self._feat_hook_handle = None

    def __del__(self):
        # 对象销毁时尽力释放 hook（训练中频繁创建/销毁实例更安全）
        self.remove_hooks()

    # ---------------- helpers ----------------
    def get_last_feat(self):
        """返回最近一次 forward 捕获到的最后阶段特征（供 Grad-CAM 等可视化使用）"""
        return self.last_feat

    def _maybe_repeat_to_3(self, x: torch.Tensor) -> torch.Tensor:
        """当 repeat_to_3ch=True 且通道为 1/2 时，将其复制到 3 通道"""
        if not self._repeat_to_3:
            return x
        c = x.size(1)
        if c == 1:
            return x.repeat(1, 3, 1, 1)
        if c == 2:
            # 2 -> 3：第三通道复制第一个
            return torch.cat([x, x[:, :1]], dim=1)
        return x

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准 forward，返回 logits: (B, num_classes)
        结尾做一次 squeeze(-1)，与您 ResNet 的行为保持一致（对 (B, C) 无影响）
        """
        x = self._maybe_repeat_to_3(x)
        logits = self.backbone(x)  # (B, num_classes)
        logits = logits.squeeze(-1)
        return logits

