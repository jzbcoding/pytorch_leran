import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transforms=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True,num_workers=2)


