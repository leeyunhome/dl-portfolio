# VGG19 Transfer Learning — CIFAR-10

→ **[View the executed notebook (HTML)](./transfer_learning_vgg19.html)**

Feature-extraction transfer learning: a pretrained VGG19 (ImageNet weights) with its
convolutional base frozen and a new classifier trained on top, to classify CIFAR-10's
10 classes.

**Source**: adapted from the [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/TableOfContents)
(MIT License, Copyright (c) 2023 Johannes Kolbe — see [../LICENSE](../LICENSE)).

**What was changed from the original**:
- Ported from a Colab-oriented notebook to run directly on a local RTX 3080 GPU server
  (PyTorch 2.9.1+cu128) — no Colab GPU allocation wait.
- Found and documented a classifier-head bug in the original tutorial: setting
  `pre_vgg19.classifier[6].out_features = NUM_CLASSES` only overwrites the attribute,
  it does not resize the actual `Linear` layer weights (stays 4096x1000). Left as-is to
  match the original tutorial, with the correct fix noted in a comment
  (`pre_vgg19.classifier[6] = nn.Linear(4096, NUM_CLASSES)`).
- Fixed the Kaggle-only inference image path to a local file.

## Setup

- Dataset: CIFAR-10 (50,000 train / 10,000 validation images, 10 classes, balanced)
- Model: `torchvision.models.vgg19(weights='IMAGENET1K_V1')`, convolutional base frozen
- Input size: 224x224 (resized from CIFAR-10's native 32x32)
- Batch size: 128, Adam optimizer (lr=1e-4), CosineAnnealingLR scheduler
