import argparse
import torch
from torch import args,nn, optim
from torchvision import datasets, transforms, models

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Path to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg11', help='Pretrained model architecture (e.g., vgg11, resnet18)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    return parser.parse_args()

def build_classifier(model, hidden_units, num_classes):
    if 'vgg' in args.arch:
        # VGG-like model architecture
        classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    elif 'resnet' in args.arch:
        # ResNet model architecture
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, num_classes),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    
def build_optimizer(model, learning_rate):
    if 'vgg' in args.arch:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif 'resnet' in args.arch:
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    return optimizer

def validate(model, validloader, criterion, device):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == labels).sum().item()
    return accuracy / len(validloader.dataset)

def main():
    args = parse_arguments()

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(args.data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(args.data_dir + '/valid', transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    if 'vgg' in args.arch:
        model = getattr(models, args.arch)(pretrained=True)
    elif 'resnet' in args.arch:
        model = getattr(models, args.arch)(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    build_classifier(model, args.hidden_units, len(train_data.class_to_idx))

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    criterion = nn.NLLLoss()
    optimizer = build_optimizer(model, args.learning_rate)
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {running_loss / len(trainloader):.4f}")

        validation_accuracy = validate(model, validloader, criterion, device)
        print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': args.epochs
    }
    torch.save(checkpoint, args.save_dir)
    print(f"Model checkpoint saved as '{args.save_dir}' with class_to_idx mapping attached to the model.")

if __name__ == '__main__':
    main()
