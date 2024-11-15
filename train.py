import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import SemanticSegmentationDataset
from src.augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
)

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

EPOCH = 200
ENCODER = "efficientnet-b0"
ENCODER_WEIGHTS = None
CLASSES = ["background", "flower", "plant"]
ACTIVATION = (
    "softmax"  # could be None for logits or 'softmax2d' for multiclass segmentation
)
DEVICE = "cuda"

model = smp.FPN(
    encoder_name=ENCODER,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)

train_dataset = SemanticSegmentationDataset(
    data_dir="./train",
    classes=["background", "flower", "plant"],
    preprocessing=get_preprocessing(preprocessing_fn),
)

valid_dataset = SemanticSegmentationDataset(
    data_dir="./valid",
    classes=["background", "flower", "plant"],
    preprocessing=get_preprocessing(preprocessing_fn),
)


set[1]

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.Fscore(),
    smp.utils.metrics.Recall(),
    smp.utils.metrics.Precision(),
]

optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=0.001),
    ]
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCH, eta_min=0.00001
)

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

min_loss = float("inf")

patient = 0
patient_limit = 30
for i in range(0, EPOCH):

    print("\nEpoch: {}".format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    if min_loss > valid_logs["dice_loss"]:
        min_loss = valid_logs["dice_loss"]
        torch.save(model, "./checkpoint/best_model_new.pth")
        print("Model saved!")
        patient = 0
    else:
        patient += 1
        if patient > patient_limit:
            print("early stopping")
            break

    scheduler.step()