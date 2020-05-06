#! /usr/bin/env python3

import os
from src import load_model_from_encoder as model_from_encoder
from src import initialize_model_from_file as model_from_file
from src import train_yolo
import torch
import torchvision
from helper import collate_fn
from data_helper import LabeledDataset
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--no_pretrain', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--filename', type=str, default='kobe_model')
parser.add_argument('--continue_training', action='store_true')
parser.add_argument('--continue_from', type=str)
opt = parser.parse_args()

print(opt)

if opt.continue_training:
    if not os.path.exists(opt.continue_from):
        print(f'Cannot continue training from {opt.continue_from}. '
              'Please set the --continue_from flag appropriately.')
        raise FileNotFoundError('--continue_from flag not set correctly')

cuda = torch.cuda.is_available()
device = 'cuda:0' if cuda else 'cpu'

if opt.no_pretrain:
    from src import KobeModel

    kobe_model = KobeModel(num_classes=10, encoder_features=6, rm_dim=800)
else:
    kobe_model = model_from_encoder('pretrain_model_2_epochs.pt')

if opt.continue_training:
    kobe_model = model_from_file(opt.continue_from)

kobe_model.to(device)

lr = opt.lr
b1 = 0.9
b2 = 0.999

kobe_optimizer = torch.optim.Adam(kobe_model.parameters(),
                                  lr=lr,
                                  betas=(b1, b2))

n_epochs = opt.n_epochs

image_folder = 'data'
annotation_csv = 'data/annotation.csv'

transform = torchvision.transforms.ToTensor()

labeled_scene_index = np.arange(106, 134)
labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index,
                                  transform=transform,
                                  extra_info=False,
                                  )
trainloader = torch.utils.data.DataLoader(labeled_trainset,
                                          batch_size=opt.batch_size,
                                          shuffle=True,
                                          num_workers=0,
                                          collate_fn=collate_fn,
                                          )


for epoch in range(n_epochs):
    print("EPOCH: {}".format(epoch))
    train_yolo(trainloader, kobe_model, kobe_optimizer, opt.verbose)

    torch.save(kobe_model.state_dict(),
               f'{opt.filename}_{epoch}_epochs.pt')
    try:
        # keep the last 3 epochs and remove any previous ones
        os.remove(f'{opt.filename}_{epoch - 3}_epochs.pt')
    except FileNotFoundError:
        pass
