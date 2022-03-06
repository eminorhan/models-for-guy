## Pretrained models for Guy's relation learning project

This repository contains pretrained models for Guy's relation learning project to test different hypotheses. All models use a `resnext50_32x4d` backbone available from `torchvision.models`.

### Temporal classification (TC) models trained with or without horizontal/vertical flip augmentations

We trained four models on headcam data from **baby S** (sampled at 1 fps; ~741K frames total) using the temporal classification objective. All models were trained for 20 epochs using the Adam optimizer with learning rate 0.0005. The models only differ in whether or not horizontal/vertical flip augmentations were used during training. These models are useful for testing the effect of horizontal/vertical flip augmentations in relational learning.

* [`TC-S-hv.tar`](https://drive.google.com/file/d/1Q5eIZyA00vSxboYC1dcb6BZa6BzP5Pe5/view?usp=sharing): model using both horizontal and vertical flip augmentations. The data augmentation pipeline used during pre-training (where `transforms` is `torchvision.transforms`):
```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

* [`TC-S-h.tar`](https://drive.google.com/file/d/1eLt-sDh3GSFDReu2KFr7Zv8xSRl_of2n/view?usp=sharing): model using only horizontal flip augmentations. The data augmentation pipeline used during pre-training:
```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

* [`TC-S-v.tar`](https://drive.google.com/file/d/1Huvc8_xB0Vd9OikJ3r6UfejPZl4b-0ef/view?usp=sharing): model using only vertical flip augmentations. The data augmentation pipeline used during pre-training:
```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

* [`TC-S.tar`](https://drive.google.com/file/d/1Yd6GqZRySDICmL1nMJ8bebtjigg3H8r0/view?usp=sharing): model not using any horizontal or vertical flip augmentations. The data augmentation pipeline used during pre-training:
```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

These pretrained models come with the temporal classification head attached and wrapped in a `DataParallel` layer, so in order to load them, please use something along the lines of:
```python
import torch
from torchvision.models as models

model = models.resnext50_32x4d(pretrained=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=2575, bias=True)
model = torch.nn.DataParallel(model).cuda()

checkpoint = torch.load('TC-S-hv.tar')
model.load_state_dict(checkpoint['model_state_dict'])
```
where `2575` is the number of temporal classes in this case.

### DINO models trained on headcam data from baby S and on ImageNet

We further trained two models using the self-supervised DINO algorithm either on the headcam data from baby S described above or on ImageNet. The models were trained identically, the only difference was the dataset used for training, hence this pair of models are useful for testing the effect of pretraining dataset on relation learning. The ImageNet model was trained for 37 epochs and the baby S model was trained for 38 epochs.

* [`DINO-S.pth`](https://drive.google.com/file/d/1FqpjQQDhKj4sCnjL-utzPuu8-tmBi5Gi/view?usp=sharing): DINO model trained on headcam data from baby S ([training log](https://github.com/eminorhan/models-for-guy/blob/master/assets/DINO-S-log.txt)).
* [`DINO-ImageNet.pth`](https://drive.google.com/file/d/1lWADEQAdTAXvLIn5Jc7OTTLwjINaXOYP/view?usp=sharing): DINO model trained on ImageNet ([training log](https://github.com/eminorhan/models-for-guy/blob/master/assets/DINO-ImageNet-log.txt)).

For loading these checkpoints, I provide a function [`load_dino_model`](https://github.com/eminorhan/models-for-guy/blob/master/load_dino_model.py) in this repository; simply use it like this:
```python
model = models.resnext50_32x4d(pretrained=False)
model = load_dino_model(model, 'DINO-S.pth', verbose=True)
```
Please let me know if you encounter any issues.
