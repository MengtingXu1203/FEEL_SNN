import torch
import torch.nn.parallel
import torch.optim
from tqdm import tqdm

from models.layers import *
from functions import make_filter, random_filter_image

def train(model, device, train_loader, criterion, optimizer, T, dvs, freq_filter=None):
    running_loss = 0
    model.train()
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)

        images = images.to(device)

        if T == 0:
            outputs = model(images)
        else:
            outputs = model(images, freq_filter)
            outputs = outputs.mean(1)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

def rat_train(model, device, train_loader, criterion, optimizer, T, atk, beta,freq_filter=None):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        images = atk(images, labels) 
        
        if T > 0:
            outputs = model(images, freq_filter).mean(1)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()

        orthogonal_retraction(model, beta) 
        convex_constraint(model)
    
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

def at_train(model, device, train_loader, criterion, optimizer, T, atk, beta,freq_filter=None):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        images = atk(images, labels) 
        
        if T > 0:
            outputs = model(images, freq_filter).mean(1)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
    
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

def orthogonal_retraction(model, beta=0.002):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if isinstance(module, nn.Conv2d):
                    weight_ = module.weight.data
                    sz = weight_.shape
                    weight_ = weight_.reshape(sz[0],-1)
                    rows = list(range(module.weight.data.shape[0]))
                elif isinstance(module, nn.Linear):
                    if module.weight.data.shape[0] < 200: # set a sample threshold for row number
                        weight_ = module.weight.data
                        sz = weight_.shape
                        weight_ = weight_.reshape(sz[0], -1)
                        rows = list(range(module.weight.data.shape[0]))
                    else:
                        rand_rows = np.random.permutation(module.weight.data.shape[0])
                        rows = rand_rows[: int(module.weight.data.shape[0] * 0.3)]
                        weight_ = module.weight.data[rows,:]
                        sz = weight_.shape
                module.weight.data[rows,:] = ((1 + beta) * weight_ - beta * weight_.matmul(weight_.t()).matmul(weight_)).reshape(sz)


def convex_constraint(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ConvexCombination):
                comb = module.comb.data
                alpha = torch.sort(comb, descending=True)[0]
                k = 1
                for j in range(1,module.n+1):
                    if (1 + j * alpha[j-1]) > torch.sum(alpha[:j]):
                        k = j
                    else:
                        break
                gamma = (torch.sum(alpha[:k]) - 1)/k
                module.comb.data -= gamma
                torch.relu_(module.comb.data)

def train_poisson(model, device, train_loader, criterion, optimizer, T):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate((train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        if T > 0:
            outputs = model(images).mean(0)
        else:
            outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
    
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

def val(model, test_loader, device, T, dvs, atk=None, freq_filter=None):
    #atk = None
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        ori = inputs
        if atk is not None:
            inputs = atk(inputs, targets.to(device))
            model.module.set_simulation_time(T)

        with torch.no_grad():
            if T > 0:
                outputs = model(inputs,freq_filter)
                outputs = outputs.mean(1) 
            else:
                outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    return final_acc