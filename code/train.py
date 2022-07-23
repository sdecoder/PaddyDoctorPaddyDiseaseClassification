import warnings
import sklearn.exceptions

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
from tqdm.auto import tqdm
from collections import defaultdict
from shutil import copyfile
import pandas as pd
import numpy as np
import os
import random
import gc
import cv2
import glob

gc.enable()
pd.set_option('display.max_columns', None)

# Visialisation
import matplotlib.pyplot as plt
# % matplotlib inline
import seaborn as sns

sns.set(style="whitegrid")

# Image Aug
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

# Deep Learning
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, label_ranking_average_precision_score as lrap
from sklearn.model_selection import train_test_split

# Random Seed Initialize
RANDOM_SEED = 42


def seed_everything(seed=RANDOM_SEED):
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True
  pass

def get_label2id():
  label2id = {'bacterial_leaf_blight': 0,
              'bacterial_leaf_streak': 1,
              'bacterial_panicle_blight': 2,
              'blast': 3,
              'brown_spot': 4,
              'dead_heart': 5,
              'downy_mildew': 6,
              'hispa': 7,
              'normal': 8,
              'tungro': 9}
  return label2id


def detect_input_data():
  print(f"[trace] detecting input data")
  csv_dir = '../data/'
  train_dir = '../data/train_images/'
  test_dir = '../data/test_images/'
  train_file_path = os.path.join(csv_dir, 'train.csv')
  print(f'Train file: {train_file_path}')
  if not os.path.exists(train_file_path):
    print(f'[trace] target file {train_file_path} does not exist, exist')
    exit(-1)

  train_df = pd.read_csv(train_file_path)
  train_df['image_path'] = train_df.apply(lambda row: train_dir + row['label'] + '/' + row['image_id'], axis=1)
  train_df['label_enc'] = train_df.apply(lambda row:  get_label2id()[row['label']], axis=1)

  print(f'[trace] train_df.head(): {train_df.head()}')
  return train_df


def make_configuration(device, train_df):
  params = {
    'model': 'efficientnet_b3',
    'fp16': True,
    'pretrained': True,
    'inp_channels': 3,
    'im_size': 300,
    'device': device,
    'lr': 5e-4,
    'weight_decay': 1e-6,
    'batch_size': 85,
    'num_workers': 0,
    'epochs': 50,
    'out_features': train_df['label'].nunique(),
    'dropout': 0.2,
    #'num_fold': train_df['kfold'].nunique(),
    'mixup': False,
    'mixup_alpha': 1.0,
    'scheduler_name': 'CosineAnnealingWarmRestarts',
    'T_0': 10,
    'T_max': 5,
    'T_mult': 1,
    'min_lr': 1e-6,
    'max_lr': 1e-3,
  }

  return params


def get_train_transforms(params):
  DIM = params['im_size']
  return albumentations.Compose(
    [
      albumentations.Resize(DIM, DIM),
      albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
      ),
      albumentations.HorizontalFlip(p=0.5),
      albumentations.VerticalFlip(p=0.5),
      albumentations.Rotate(limit=180, p=0.7),
      albumentations.RandomResizedCrop(
        height=DIM, width=DIM, scale=(0.8, 1.0), p=0.5
      ),
      albumentations.Cutout(
        num_holes=15, max_h_size=30, max_w_size=30,
        fill_value=0, always_apply=False, p=0.5
      ),
      albumentations.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5
      ),
      albumentations.HueSaturationValue(
        hue_shift_limit=0.2, sat_shift_limit=0.2,
        val_shift_limit=0.2, p=0.5
      ),
      ToTensorV2(p=1.0),
    ]
  )


def mixup_data(x, y, params):
  assert params['mixup_alpha'] > 0
  assert x.size(0) > 1

  if params['mixup_alpha'] > 0:
    lam = np.random.beta(
      params['mixup_alpha'], params['mixup_alpha']
    )
  else:
    lam = 1

  batch_size = x.size()[0]
  if params['device'].type == 'cuda':
    index = torch.randperm(batch_size).cuda()
  else:
    index = torch.randperm(batch_size)

  mixed_x = lam * x + (1 - lam) * x[index, :]
  y_a, y_b = y, y[index]
  return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
  return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_valid_transforms(params):

  DIM = params['im_size']
  return albumentations.Compose(
    [
      albumentations.Resize(DIM, DIM),
      albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
      ),
      ToTensorV2(p=1.0)
    ]
  )

#define the dataset
class PaddyDataset(Dataset):
  def __init__(self, images_filepaths, targets, transform=None):
    self.images_filepaths = images_filepaths
    self.targets = targets
    self.transform = transform

  def __len__(self):
    return len(self.images_filepaths)

  def __getitem__(self, idx):
    image_filepath = self.images_filepaths[idx]
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if self.transform is not None:
      image = self.transform(image=image)['image']

    label = torch.tensor(self.targets[idx]).long()
    return image, label

def show_image(train_dataset, id2label, inline=4,):

  plt.figure(figsize=(20, 10))
  for i in range(inline):
    rand = random.randint(0, len(train_dataset))
    image, label = train_dataset[rand]
    plt.subplot(1, inline, i % inline + 1)
    plt.axis('off')
    plt.imshow(image.permute(2, 1, 0))
    plt.title(f'Label: {id2label[label.item()]}')


def visual_examples(train_df, params):

  X_train = train_df['image_path']
  y_train = train_df['label_enc']

  # Pytorch Dataset Creation
  train_dataset = PaddyDataset(
    images_filepaths=X_train.values,
    targets=y_train.values,
    transform=get_train_transforms(params)
  )

  return train_dataset

def get_scheduler(optimizer, scheduler_params, train_df):

  if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
    scheduler = CosineAnnealingWarmRestarts(
      optimizer,
      T_0=scheduler_params['T_0'],
      T_mult=scheduler_params['T_mult'],
      eta_min=scheduler_params['min_lr'],
      last_epoch=-1
    )

    '''
      elif scheduler_params['scheduler_name'] == 'OneCycleLR':
    scheduler = OneCycleLR(
      optimizer,
      max_lr=scheduler_params['max_lr'],
      steps_per_epoch=int(((scheduler_params['num_fold'] - 1) * train_df.shape[0]) / (
            scheduler_params['num_fold'] * scheduler_params['batch_size'])) + 1,
      epochs=scheduler_params['epochs'],
    )
    '''

  elif scheduler_params['scheduler_name'] == 'CosineAnnealingLR':
    scheduler = CosineAnnealingLR(
      optimizer,
      T_max=scheduler_params['T_max'],
      eta_min=scheduler_params['min_lr'],
      last_epoch=-1
    )
  return scheduler

class MetricMonitor:
  def __init__(self, float_precision=3):
    self.float_precision = float_precision
    self.reset()

  def reset(self):
    self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

  def update(self, metric_name, val):
    metric = self.metrics[metric_name]

    metric["val"] += val
    metric["count"] += 1
    metric["avg"] = metric["val"] / metric["count"]

  def __str__(self):
    return " | ".join(
      [
        "{metric_name}: {avg:.{float_precision}f}".format(
          metric_name=metric_name, avg=metric["avg"],
          float_precision=self.float_precision
        )
        for (metric_name, metric) in self.metrics.items()
      ]
    )

def usr_acc_score(output, target):
  y_pred = output.softmax(dim=1).argmax(dim=1).detach().cpu()
  target = target.cpu()
  return accuracy_score(target, y_pred)

def f1(output, target):
  y_pred = output.softmax(dim=1).argmax(dim=1).detach().cpu()
  target = target.cpu()
  return f1_score(target, y_pred, average='weighted')

def precision(output, target):
  y_pred = output.softmax(dim=1).argmax(dim=1).detach().cpu()
  target = target.cpu()
  return precision_score(target, y_pred, average='weighted')

def recall(output, target):
  y_pred = output.softmax(dim=1).argmax(dim=1).detach().cpu()
  target = target.cpu()
  return recall_score(target, y_pred, average='weighted')

def usr_roc_score(output, target):
  y_pred = output.softmax(dim=1).argmax(dim=1).detach().cpu()
  target = target.cpu()
  return roc_auc_score(target, y_pred, average='weighted', multi_class='ovr')

class PaddyNet(nn.Module):
  def __init__(self, params):

    super().__init__()

    model_name = params['model']
    out_features = params['out_features']
    inp_channels = params['inp_channels']
    pretrained = params['pretrained']
    self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels)
    out_channels = self.model.conv_stem.out_channels
    kernel_size = self.model.conv_stem.kernel_size
    stride = self.model.conv_stem.stride
    padding = self.model.conv_stem.padding
    bias = self.model.conv_stem.bias
    self.model.conv_stem = nn.Conv2d(inp_channels, out_channels,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=bias)
    n_features = self.model.classifier.in_features
    self.model.classifier = nn.Identity()
    self.dropout = nn.Dropout(params['dropout'])
    self.fc = nn.Linear(n_features, out_features)

  def forward(self, image):
    embeddings = self.model(image)
    x = self.dropout(embeddings)
    output = self.fc(x)
    return output

def train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler=None, scaler=None):
  metric_monitor = MetricMonitor()
  model.train()
  stream = tqdm(train_loader)

  for i, (images, target) in enumerate(stream, start=1):
    if params['mixup']:
      images, target_a, target_b, lam = mixup_data(images, target, params)
      images = images.to(params['device'])
      target_a = target_a.to(params['device'])
      target_b = target_b.to(params['device'])
    else:
      images = images.to(params['device'], non_blocking=True)
      target = target.to(params['device'], non_blocking=True)

    if params['fp16']:
      with torch.cuda.amp.autocast():
        output = model(images)
        if params['mixup']:
          loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
          loss = criterion(output, target)
    else:
      output = model(images)
      if params['mixup']:
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
      else:
        loss = criterion(output, target)

    metric_monitor.update('Loss', loss.item())
    f1_batch = f1(output, target)
    metric_monitor.update('F1', f1_batch)
    precision_batch = precision(output, target)
    metric_monitor.update('Precision', precision_batch)
    recall_batch = recall(output, target)
    metric_monitor.update('Recall', recall_batch)
    acc_batch = usr_acc_score(output, target)
    metric_monitor.update('Accuracy', acc_batch)

    if params['fp16']:
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      optimizer.step()

    if scheduler is not None:
      scheduler.step()

    optimizer.zero_grad()
    stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")

def validate_fn(val_loader, model, criterion, epoch, params):
  metric_monitor = MetricMonitor()
  model.eval()
  stream = tqdm(val_loader)
  final_targets = []
  final_outputs = []
  with torch.no_grad():
    for i, (images, target) in enumerate(stream, start=1):
      images = images.to(params['device'], non_blocking=True)
      target = target.to(params['device'], non_blocking=True)
      output = model(images)
      loss = criterion(output, target)
      metric_monitor.update('Loss', loss.item())
      f1_batch = f1(output, target)
      metric_monitor.update('F1', f1_batch)
      precision_batch = precision(output, target)
      metric_monitor.update('Precision', precision_batch)
      recall_batch = recall(output, target)
      metric_monitor.update('Recall', recall_batch)
      acc_batch = usr_acc_score(output, target)
      metric_monitor.update('Accuracy', acc_batch)
      stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")

      targets = target.detach().cpu().numpy().tolist()
      outputs = output.softmax(dim=1).argmax(dim=1).detach().cpu().numpy().tolist()

      final_targets.extend(targets)
      final_outputs.extend(outputs)
  return final_outputs, final_targets

def train_validate(train_df, params):

  print(f'[trace] train_validate(train_df, params)')
  best_models_of_each_fold = []
  accuracy_tracker = []

  # Data Split to train and Validation
  train, valid = train_test_split(train_df, test_size=0.2)

  X_train = train['image_path']
  y_train = train['label_enc']
  X_valid = valid['image_path']
  y_valid = valid['label_enc']

  # Pytorch Dataset Creation
  train_dataset = PaddyDataset(
    images_filepaths=X_train.values,
    targets=y_train.values,
    transform=get_train_transforms(params)
  )

  valid_dataset = PaddyDataset(
    images_filepaths=X_valid.values,
    targets=y_valid.values,
    transform=get_valid_transforms(params)
  )

  # Pytorch Dataloader creation
  train_loader = DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=True,
    num_workers=params['num_workers'], pin_memory=True
  )

  val_loader = DataLoader(
    valid_dataset, batch_size=params['batch_size'], shuffle=False,
    num_workers=params['num_workers'], pin_memory=True
  )

  # Model, cost function and optimizer instancing
  model = PaddyNet(params)
  model = model.to(params['device'])
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                weight_decay=params['weight_decay'],
                                amsgrad=False)
  scheduler = get_scheduler(optimizer, params, train_df)

  if params['fp16']:
    scaler = torch.cuda.amp.GradScaler()
  else:
    scaler = None

  # Training and Validation Loop
  best_accracy = -np.inf
  best_epoch = np.inf
  best_model_name = None
  for epoch in range(1, params['epochs'] + 1):
    print(f'[trace] current working epoch: {epoch}')
    train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler, scaler)
    predictions, valid_targets = validate_fn(val_loader, model, criterion, epoch, params)
    accuracy = round(accuracy_score(valid_targets, predictions), 3)
    print(f'[trace] epoch {epoch} done with accuracy: {accuracy}')

    if accuracy > best_accracy:
      best_accracy = accuracy
      best_epoch = epoch
      if best_model_name is not None:
        os.remove(best_model_name)
      torch.save(model.state_dict(), f"{params['model']}_{epoch}_epoch_{accuracy}_accuracy.pth")
      best_model_name = f"{params['model']}_{epoch}_epoch_{accuracy}_accuracy.pth"

  # Print summary of this fold
  print('')
  print(f'The best Accuracy: {best_accracy} was achieved on epoch: {best_epoch}.')
  print(f'The Best saved model is: {best_model_name}')
  best_models_of_each_fold.append(best_model_name)
  accuracy_tracker.append(best_accracy)
  print(''.join(['#'] * 50))
  del model
  gc.collect()
  torch.cuda.empty_cache()

  print('')
  print(f'Average Accuracy of all folds: {round(np.mean(accuracy_tracker), 4)}')

  pass

def main():
  print(f'[trace] working in the main function body')
  # Device Optimization
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  print(f'Using device: {device}')
  seed_everything()
  train_df = detect_input_data()

  '''
  label2id = get_label2id()
  id2label = {v: k for k, v in label2id.items()}  
  '''
  params = make_configuration(device, train_df=train_df)
  albumentations = get_train_transforms(params)
  #train_dataset = visual_examples(train_df)
  train_validate(train_df, params)
  pass


if __name__ == '__main__':
  main()
  pass
