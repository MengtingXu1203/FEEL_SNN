import torch
import torch.nn as nn
from torchattacks.attack import Attack
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import math
import numpy as np
import cv2
import seaborn as sns


class PGD(Attack):
    r"""
    altered from torchattack
    """
    def __init__(self, model, forward_function=None, eps=0.3,
                 alpha=2/255, steps=40, random_start=True, T=None, **kwargs):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.forward_function = forward_function
        self._targeted = False
        self.T = T

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        self.model.eval()
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            if self.forward_function is not None:
                if self.T >0:
                    outputs = self.forward_function(self.model, adv_images, self.T)
                else:
                    outputs = self.model(adv_images)
            else:
                outputs = self.model(adv_images)

            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
 

        return adv_images
