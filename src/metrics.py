import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips as lpips_lib
from typing import Dict


class MetricsEvaluator:
    def __init__(self, device: torch.device):
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)  # [-1,1] range => 2.0
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.lpips = lpips_lib.LPIPS(net='vgg').to(device)

    @torch.no_grad()
    def evaluate_batch(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        # preds/targets in [-1,1]
        psnr_val = self.psnr(preds, targets).item()
        ssim_val = self.ssim(preds, targets).item()
        # LPIPS expects [0,1] normalized RGB tensors
        preds_01 = (preds.clamp(-1, 1) + 1) / 2
        targets_01 = (targets.clamp(-1, 1) + 1) / 2
        lpips_val = self.lpips(preds_01, targets_01).mean().item()
        return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}

