import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from VGGNet import VGGNet

#https://dgschwend.github.io/netscope/#/preset/vgg-16

if __name__ == '__main__':  
    torch.multiprocessing.freeze_support()

    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    train_set = datasets.CIFAR100(root='data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = datasets.CIFAR100(root='data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    # print(device)
    Net = VGGNet().to(device) #实例化网络模型
    Net.load_state_dict(torch.load("model/model6.pth", map_location=device, weights_only=True))
    Net.eval()

    lossFn = torch.nn.CrossEntropyLoss() #交叉熵损失函数
    optimizer = torch.optim.SGD(Net.parameters(), lr=0.02)

    init = 6
    writer = SummaryWriter(log_dir='logs')
    epochs = 20
    for epoch in range(epochs):
        print(f"The Result of Round {epoch + 1} : ")
        train_loss = 0.0
        test_loss = 0.0
        train_acc = 0
        test_acc = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() #清空梯度
            outputs = Net(images)

            loss = lossFn(outputs, labels)
            loss.backward() #反向传播
            optimizer.step()

            _, index = torch.max(outputs, 1)

            train_loss += loss.item()
            train_acc += torch.sum(index == labels).item()
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = Net(images)

                loss = lossFn(outputs, labels)

                _, index = torch.max(outputs, 1)
                test_loss += loss.item()
                test_acc += torch.sum(index == labels).item()
        print(f"Train Loss: {train_loss / len(trainloader)}")
        print(f"Train Accuracy: {train_acc / len(trainloader.dataset)}")
        print(f"Test Loss: {test_loss / len(testloader)}")
        print(f"Test Accuracy: {test_acc / len(testloader.dataset)}")
        writer.add_scalar('loss/train', train_loss / len(trainloader), epoch+1)
        writer.add_scalar('acc/train', train_acc / len(trainloader.dataset), epoch+1)
        writer.add_scalar('loss/test', test_loss / len(testloader), epoch+1)
        writer.add_scalar('acc/test', test_acc / len(testloader.dataset), epoch+1)
        if (epoch + 1) % 10 == 0:
            torch.save(Net.state_dict(), f'model/model{int((epoch + 1) / 10) + init}.pth')