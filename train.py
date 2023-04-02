"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import os
import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import config
from models import WSDAN
from datasets import get_trainval_datasets
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment
import matplotlib.pyplot as plt
import  gc

gc.collect()
torch.cuda.empty_cache()

# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
# General loss functions
import seaborn as sns
import torch
import torch.nn as nn
import torch
import numpy as np


def cutmix(batch, alpha):
    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''MixUp 数据增强方法'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, targets, reduction=self.reduction)
        return self.alpha * loss



cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
crop_metric = TopKAccuracyMetric(topk=(1, 5))
drop_metric = TopKAccuracyMetric(topk=(1, 5))

EPOCHS=config.epochs
NUM_STEPS_PER_EPOCH = 1000  # Number of steps per epoch
TOTAL_STEPS = EPOCHS * NUM_STEPS_PER_EPOCH  # Total number of steps

def main():
    # 初始化全局变量
    global train_losses, val_losses, train_accuracies, val_accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    global y_true_global, y_pred_global, y_pred_prob_global
    y_true_global = []
    y_pred_global = []
    y_pred_prob_global = []

    ##################################
    # Initialize saving directory
    ##################################
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    ##################################
    # Logging setting
    ##################################
    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Load dataset
    ##################################
    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
                                               num_workers=config.workers, pin_memory=True)
    num_classes = train_dataset.num_classes

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0
    net = WSDAN(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True)

    # feature_center: size of (#classes, #attention_maps * #channel_features)
    feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).to(device)

    if config.ckpt:
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(device)
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    ##################################
    # Use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ##################################
    # Optimizer, LR Scheduler
    ##################################
    learning_rate = logs['lr'] if 'lr' in logs else config.learning_rate

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # ModelCheckpoint
    ##################################
    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    ##################################
    # TRAINING
    ##################################
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
    logging.info('')



    for epoch in range(start_epoch, config.epochs):
        callback.on_epoch_begin()

        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

        train(logs=logs,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              optimizer=optimizer,
              pbar=pbar)
        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        else:
            scheduler.step()

        callback.on_epoch_end(logs, net, feature_center=feature_center)
        pbar.close()

def plot_losses_and_accuracies(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    # 绘制损失图像
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # 绘制准确率图像
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # 保存并显示图像
    plt.savefig(os.path.join(config.save_dir, 'losses_accuracies_plot.png'))
    plt.show()




def train(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()

    # begin training
    start_time = time.time()
    net.train()
    running_loss = 0.0
    running_corrects = 0
    for i, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()

        # obtain data for training
        X = X.to(device)
        y = y.to(device)
        # #mixup
        # X, targets_a, targets_b, lam = mixup_data(X, y, alpha=1.0, use_cuda=device.type == 'cuda')

        # X, y = cutmix((X, y), alpha=config.cutmix_alpha)
        # X = X.to(device)
        # y_orig, y_shuffled, lam = y
        # y_orig = y_orig.to(device)
        # y_shuffled = y_shuffled.to(device)

        ##################################
        # Raw Image
        ##################################
        # raw images forward
        y_pred_raw, feature_matrix, attention_map = net(X)

        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)

        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)

        ##################################
        # Attention Cropping
        ##################################
        with torch.no_grad():
            crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)

        # crop images forward
        y_pred_crop, _, _ = net(crop_images)

        ##################################
        # Attention Dropping
        ##################################
        with torch.no_grad():
            drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))



        # drop images forward
        y_pred_drop, _, _ = net(drop_images)
        gc.collect()
        torch.cuda.empty_cache()

        focal_loss = FocalLoss()
        label_smoothing_loss = LabelSmoothingLoss(classes=200, smoothing=0.1)

        # # focal_loss
        # batch_loss = focal_loss(y_pred_raw, y) / 3. + \
        #              focal_loss(y_pred_crop, y) / 3. + \
        #              focal_loss(y_pred_drop, y) / 3. + \
        #              center_loss(feature_matrix, feature_center_batch)

        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_crop, y) / 3. + \
                     cross_entropy_loss(y_pred_drop, y) / 3. + \
                     center_loss(feature_matrix, feature_center_batch)

        # #smoothing_loss
        # batch_loss = label_smoothing_loss(y_pred_raw, y) / 3. + \
        #              label_smoothing_loss(y_pred_crop, y) / 3. + \
        #              label_smoothing_loss(y_pred_drop, y) / 3. + \
        #              center_loss(feature_matrix, feature_center_batch)
        #mixup
        # batch_loss = mixup_criterion(cross_entropy_loss, y_pred_raw, targets_a, targets_b, lam) / 3. + \
        #              mixup_criterion(cross_entropy_loss, y_pred_crop, targets_a, targets_b, lam) / 3. + \
        #              mixup_criterion(cross_entropy_loss, y_pred_drop, targets_a, targets_b, lam) / 3. + \
        #              center_loss(feature_matrix, feature_center_batch)

        #cutmix
        # raw_loss = lam * cross_entropy_loss(y_pred_raw, y_orig) + (1 - lam) * cross_entropy_loss(y_pred_raw, y_shuffled)
        # crop_loss = lam * cross_entropy_loss(y_pred_crop, y_orig) + (1 - lam) * cross_entropy_loss(y_pred_crop,
        #                                                                                            y_shuffled)
        # drop_loss = lam * cross_entropy_loss(y_pred_drop, y_orig) + (1 - lam) * cross_entropy_loss(y_pred_drop,
        #                                                                                            y_shuffled)
        #
        # batch_loss = raw_loss / 3. + crop_loss / 3. + drop_loss / 3. + center_loss(feature_matrix, feature_center_batch)

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_crop, y)
            epoch_drop_acc = drop_metric(y_pred_drop, y)

            # #mixup
            # epoch_raw_acc = raw_metric(y_pred_raw, y_orig)
            # epoch_crop_acc = crop_metric(y_pred_crop, y_orig)
            # epoch_drop_acc = drop_metric(y_pred_drop, y_orig)

        # end of this batch
        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Crop Acc ({:.2f}, {:.2f}), Drop Acc ({:.2f}, {:.2f})'.format(
            epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
            epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1])
        pbar.update()
        pbar.set_postfix_str(batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # 添加对 train_losses 和 train_accuracies 的更新
    global train_losses, train_accuracies
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_raw_acc)

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))





def validate(**kwargs):


    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            # obtain data
            X = X.to(device)
            y = y.to(device)

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, _, attention_map = net(X)

            ##################################
            # Object Localization and Refinement
            ##################################
            crop_images = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _, _ = net(crop_images)

            ##################################
            # Final prediction
            ##################################
            y_pred = (y_pred_raw + y_pred_crop) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)

    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()

    # 添加对 val_losses 和 val_accuracies 的更新
    global val_losses, val_accuracies
    val_losses.append(epoch_loss)
    val_accuracies.append(epoch_acc[0])
    global y_true_global, y_pred_global, y_pred_prob_global
    y_true_global.extend(y.cpu().numpy())
    y_pred_global.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
    y_pred_prob_global.extend(F.softmax(y_pred, dim=1).cpu().numpy())

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    # write log for this epoch
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')




if __name__ == '__main__':
    main()
    plot_losses_and_accuracies(train_losses, val_losses, train_accuracies, val_accuracies)
