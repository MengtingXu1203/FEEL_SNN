import torch
import torch.nn as nn
from torchattacks.attack import Attack


class FGSM(Attack):
    r"""
    altered from torchattack
    """
    def __init__(self, model, forward_function=None, eps=0.007, T=None, **kwargs):
        super().__init__("FGSM", model)
        self.eps = eps
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

        images.requires_grad = True
        if self.forward_function is not None:
            if self.T >0 :
                outputs = self.forward_function(self.model, images, self.T)
            else:
                outputs = self.model(images)
        else:
            outputs = self.model(images)

        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images