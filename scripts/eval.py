import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from torchvision.transforms import v2
from PIL import Image

from models.unet import UNet


def build_transforms(image_size: int):
    base = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    norm_sar = v2.Normalize(mean=[0.5], std=[0.5])
    denorm_rgb = lambda t: (t.clamp(-1, 1) + 1) / 2
    return base, norm_sar, denorm_rgb


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate UNet on a single SAR image')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--output', type=str, default='prediction.png')
    p.add_argument('--image-size', type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=3).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path
    state = torch.load(ckpt_path, map_location=device)
    if 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model.eval()

    base_t, norm_sar, denorm_rgb = build_transforms(args.image_size)

    img = Image.open(args.input).convert('L')
    x = base_t(img)
    x = norm_sar(x).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)
        y = denorm_rgb(y).squeeze(0).cpu()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    y_img = (y.permute(1, 2, 0).numpy() * 255).astype('uint8')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(y_img).save(out_path)


if __name__ == '__main__':
    main()

