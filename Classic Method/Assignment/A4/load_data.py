import torch
import torchvision
import torchvision.transforms as transforms



def get_data():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

    #Downloading test data
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)

    return trainloader, testloader
