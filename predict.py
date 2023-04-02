import os
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets
import config
from utils import get_transform
from datasets.bird_dataset import image_label
from models import WSDAN
from datasets import get_trainval_datasets
from utils import TopKAccuracyMetric, batch_augment

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# visualize
visualize = config.visualize
savepath = config.eval_savepath
if visualize:
    os.makedirs(savepath, exist_ok=True)

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)


def main():
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    try:
        ckpt = config.eval_ckpt
    except:
        logging.info('Set ckpt for evaluation in config.py')
        return

    ##################################
    # Dataset for testing
    ##################################
    pred_dataset=datasets.ImageFolder('C:/Users/ZYYZNB/bishe_3/predict',transform=get_transform(config.image_size,phase='val'))
    pred_loader = DataLoader(pred_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    net = WSDAN(num_classes=200, M=config.num_attentions, net=config.net)


    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))

    ##################################
    # use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(pred_loader), unit=' batches')
        pbar.set_description('Validation')
        for i, (X, y) in enumerate(pred_loader):
            X = X.to(device)
            y = y.to(device)

            # WS-DAN
            y_pred_raw, _, attention_maps = net(X)

            # Augmentation with crop_mask
            crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)

            y_pred_crop, _, _ = net(crop_image)

            y_pred = (y_pred_raw + y_pred_crop) / 2.
            pred_ace, pred = y_pred.cpu().topk(1, 1, True, True)
            print(pred_ace, pred.item())
            b = pred.item() + 1
            c = str(b)
            print(c)



        pbar.close()

if __name__ == '__main__':
    main()