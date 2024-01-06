import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# Define command-line arguments
parser = argparse.ArgumentParser(description='Predict the class of an input image')
parser.add_argument('input', help='Path to the input image')
parser.add_argument('checkpoint', help='Path to the model checkpoint')
parser.add_argument('--top_k', type=int, default=1, help='Return the top K most likely classes')
parser.add_argument('--category_names', default='cat_to_name.json', help='File mapping categories to names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
args = parser.parse_args()

# Load the checkpoint
checkpoint = torch.load(args.checkpoint)

# Load a pre-trained model (supports vgg11, vgg13, and resnet18)
if checkpoint['arch'] == 'vgg11':
    model = models.vgg11(pretrained=True)
elif checkpoint['arch'] == 'vgg13':
    model = models.vgg13(pretrained=True)
elif checkpoint['arch'] == 'resnet18':
    model = models.resnet18(pretrained=True)
else:
    raise ValueError("Unsupported model architecture")

# Replace the classifier with the one from the checkpoint
model.classifier = checkpoint['classifier']
model.load_state_dict(checkpoint['model_state_dict'])
model.class_to_idx = checkpoint['class_to_idx']

# Use GPU if available
device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Preprocess the input image
def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image

image = process_image(args.input).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    output = model(image)

# Get top K predictions and their class indices
probabilities, indices = torch.topk(output, args.top_k)
probabilities = probabilities.exp().cpu().numpy()[0]
indices = indices.cpu().numpy()[0]

# Load category names mapping
import json
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f, strict = False)

# Create a dictionary mapping from indices to classes
idx_to_class = {x: y for y, x in model.class_to_idx.items()}

# Get top K class labels
top_classes = [idx_to_class[x] for x in indices]

# Print the top K predictions with class labels
for i in range(args.top_k):
    print(f"Top-{i+1} Prediction: {top_classes[i]} (Class Index: {indices[i]}), Probability: {probabilities[i]:.3f}")

