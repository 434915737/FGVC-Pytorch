import os
import logging
import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import config
from models import WSDAN
from datasets import get_trainval_datasets
from utils import TopKAccuracyMetric, batch_augment
import random
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# GPU settings
assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Modify this list to include the paths to your three ckpt files
ckpt_list = [config.eval_ckpt1, config.eval_ckpt2, config.eval_ckpt3]



# Bagging ensemble
def bagging_ensemble_predictions(models, X, num_samples=10):
    preds = []
    for net in models:
        y_preds = []
        for _ in range(num_samples):
            idx = random.choices(range(X.shape[0]), k=X.shape[0])
            X_sample = X[idx, :]
            y_pred_raw, _, attention_maps = net(X_sample)
            crop_image = batch_augment(X_sample, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)
            y_pred_crop, _, _ = net(crop_image)
            y_pred = (y_pred_raw + y_pred_crop) / 2
            y_preds.append(y_pred)
        preds.append(torch.mean(torch.stack(y_preds), dim=0))
    return sum(preds) / len(preds)

# Boosting ensemble
def boosting_ensemble_predictions(models, X, y_true):
    # Convert the one-hot predictions to class labels
    def to_class_labels(y_true):
        return torch.argmax(y_true, dim=1).cpu().numpy()

    base_learner = DecisionTreeClassifier(max_depth=3)
    clf = AdaBoostClassifier(base_estimator=base_learner, n_estimators=len(models), algorithm='SAMME.R')

    # Extract features for each model
    features = []
    labels = to_class_labels(y_true)
    for net in models:
        y_pred_raw, _, attention_maps = net(X)
        crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop, _, _ = net(crop_image)
        y_pred = (y_pred_raw + y_pred_crop) / 2
        features.append(to_class_labels(y_pred))
    features = np.column_stack(features)

    # Train the AdaBoost classifier
    clf.fit(features, labels)

    # Make predictions
    y_pred_boosted = clf.predict(features)
    return torch.tensor(y_pred_boosted, device=y_true.device)


def ensemble_predictions(models, X):
    preds = []
    for net in models:
        y_pred_raw, _, attention_maps = net(X)
        crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)
        y_pred_crop, _, _ = net(crop_image)
        y_pred = (y_pred_raw + y_pred_crop) / 2
        preds.append(y_pred)
    return sum(preds) / len(preds)

def main():
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    # Load test dataset
    _, test_dataset = get_trainval_datasets(config.tag, resize=config.image_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Load the three models
    models = []
    for ckpt in ckpt_list:
        net = WSDAN(num_classes=test_dataset.num_classes, M=config.num_attentions, net=config.net)
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(ckpt))
        net.to(device)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.eval()
        models.append(net)

    # Top K accuracy metric
    ref_accuracy = TopKAccuracyMetric(topk=(1, 5))
    ref_accuracy.reset()

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        for i, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            #y_pred = ensemble_predictions(models, X)
            #y_pred = bagging_ensemble_predictions(models, X)  # with this line for Bagging-91.22
            y_pred = bagging_ensemble_predictions(models, X)  # or this line for Boosting-91.37

            # Top K
            epoch_ref_acc = ref_accuracy(y_pred, y)

            # end of this batch
            batch_info = 'Val Acc: Refine ({:.2f}, {:.2f})'.format(epoch_ref_acc[0], epoch_ref_acc[1])
            pbar.update()
            pbar.set_postfix_str(batch_info)

        pbar.close()

if __name__ == '__main__':
    main()
