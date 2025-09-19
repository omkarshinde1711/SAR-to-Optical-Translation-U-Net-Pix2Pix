import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import v2

from models.unet import UNet


st.set_page_config(page_title='SAR Colorization (UNet)', layout='centered')
st.title('SAR -> RGB Colorization')

default_ckpt = PROJECT_ROOT / 'results' / 'unet' / 'checkpoints' / 'epoch_5.pt'
checkpoint_path = st.text_input('Checkpoint path', str(default_ckpt))
image_size = st.number_input('Image size', min_value=64, max_value=1024, value=256, step=32)

uploaded = st.file_uploader('Upload SAR image (PNG)', type=['png'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(ckpt_path: str, in_ch: int = 1, out_ch: int = 3):
    model = UNet(in_channels=in_ch, out_channels=out_ch).to(device)
    ckpt = Path(ckpt_path)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()
    return model


if st.button('Run Colorization') and uploaded is not None and checkpoint_path:
    model = load_model(checkpoint_path)
    base_t = v2.Compose([v2.Resize((image_size, image_size)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    norm_sar = v2.Normalize(mean=[0.5], std=[0.5])
    denorm_rgb = lambda t: (t.clamp(-1, 1) + 1) / 2

    img = Image.open(uploaded).convert('L')
    x = base_t(img)
    x = norm_sar(x).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x)
        y = denorm_rgb(y).squeeze(0).cpu().permute(1, 2, 0).numpy()
    st.image(y, caption='Predicted RGB')

st.caption('Tip: Use uv to run: uv run streamlit run scripts/ui.py')

