import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from datasets.mnist import trainloader, testloader
from models.mnist import *

models = [ConvolutionalClassificator, SwinS2MLPClassificatorSmall, SwinS2MLPClassificatorSmall, SwinS2MLPClassificatorLarge]
num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for model in models:
    model = model()
    print(f"{type(model)}: {sum(p.numel() for p in model.parameters())} params.")

def main():
    for model_iter, model_class in enumerate(models):
        model = model_class()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        loss_fn = loss_fn.to(device)
        loss_log = []
        print(f"training {model_class.__name__}...")
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            loss_log.append(loss.item())
            print(f"iter {i}, epoch {epoch}/{num_epochs}, loss {loss.item():.8f}")
            
        # test
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"{model_class.__name__}: accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
