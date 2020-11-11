import torch
import torchvision
import torchvision.transforms as transforms

from model import BuddingTree

trainset = torchvision.datasets.CIFAR10(root="./data", download=True, train=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10(root="./data", download=True, train=False, transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)

LR = 0.001
SIZE = 3072
tree = BuddingTree(SIZE, 10, 5, 0.8)
optimizer = torch.optim.Adam(lr=LR, params=tree.parameters(), amsgrad=True)
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(lr=0.001, params=tree.parameters())
for e in range(200):
    epoch_loss = 0.0
    for i, (x, y) in enumerate(loader):
        y_pred = tree(x.reshape(-1, SIZE))
        loss = criterion(y_pred, y)
        epoch_loss += loss.item()
        if torch.isnan(loss):
            print(tree(x))
            for p in tree.parameters():
                print(p)
                print(p.grad)
            exit()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # tree.clamp_gamma()
            changed = tree.update_nodes()
            if changed:
                optimizer = torch.optim.Adam(lr=LR, params=tree.parameters(), amsgrad=True)
                # optimizer = torch.optim.SGD(lr=0.001, params=tree.parameters())
                print("Tree updated at iteration %d." % (i+1))
                tree.print_tree()
    x, y = iter(testloader).next()
    with torch.no_grad():
        y_pred = tree(x.reshape(-1, SIZE))
        acc = (y_pred.argmax(dim=1) == y).sum().float() / 1000
    print("Epoch: %d loss: %.5f acc: %.5f" % (e+1, epoch_loss/(i+1), acc))
    tree.print_tree()
