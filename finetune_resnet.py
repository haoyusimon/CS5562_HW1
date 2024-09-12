from resnet_attack_todo import ResnetPGDAttacker
from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from tqdm import tqdm



SEED = 1234
NUM_EPOCHS = 20
BATCH_SIZE = 100 # max for 12Gb GPU memory
LEARNING_RATE = 1e-4
NUM_IMAGES_TRAIN = 50000 # 50000 clean images + 50000 adv_images
NUM_IMAGES_EVAL = 10000
EPS = 8/255 # default eps used in the provided code
ALPHA = 2/255 # default
STEPS = 20

torch.manual_seed(SEED)

print('Loading model...')

# Initialize the original model to be attacked and the model to be finetuned
orig_model = resnet50(weights=ResNet50_Weights.DEFAULT)

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
preprocess = weights.transforms()


# Load and preprocess data
print('Loading data...')

# Load ImageNet-1k training dataset from Huggingface
train_ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, use_auth_token=True)

def preprocess_img(example):
    example['image'] = preprocess(example['image'])
    return example


# Filter out grayscale images
train_ds = train_ds.filter(lambda example: example['image'].mode == 'RGB')
# Preprocess function will be applied to images on-the-fly whenever they are being accessed in the loop
train_ds = train_ds.map(preprocess_img)
train_ds = train_ds.shuffle(seed=SEED)
# Only take desired portion of dataset
train_ds = train_ds.take(NUM_IMAGES_TRAIN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Load ImageNet-1k validation dataset from Huggingface
val_ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True, use_auth_token=True)

# Filter out grayscale images
val_ds = val_ds.filter(lambda example: example['image'].mode == 'RGB')
# Preprocess function will be applied to images on-the-fly whenever they are being accessed in the loop
val_ds = val_ds.map(preprocess_img)
val_ds = val_ds.shuffle(seed=SEED)
# Only take desired portion of dataset
val_ds = val_ds.take(NUM_IMAGES_EVAL)

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps") # for Nvidia GPU or Apple Silicon
model = model.to(device)
orig_model = orig_model.to(device)

# Initialize PGDAttacker for making adversarial images on-the-fly during finetuning
adv_model = ResnetPGDAttacker(orig_model, train_loader)
adv_model.eps = EPS 
adv_model.alpha = ALPHA
adv_model.steps = STEPS

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # train
    for images_labels in tqdm(train_loader, desc=f"Train: Epoch {epoch + 1}/{NUM_EPOCHS}", total=NUM_IMAGES_TRAIN/BATCH_SIZE):
        images, labels = images_labels['image'].to(device), images_labels['label'].to(device)
        adv_images = adv_model.pgd_attack(images, labels).to(device) # produce adv_images using PGDAttacker

        optimizer.zero_grad()
        
        outputs = model(torch.cat([images, adv_images])) # mix adv_images with original images for finetuning, hopefully can maintain the accurarcy of the finetuned model on clean images
        loss = loss_fn(outputs, torch.cat([labels, labels]))
        
        loss.backward()
        optimizer.step()
        
        # accumulate loss and accuracy
        running_loss += loss.item()
    
    # save all weights, choose the one with highest validation accuracy later
    torch.save(model.state_dict(), f"weights/epoch_{epoch}.pth")

    # validate
    for images, labels in tqdm(val_loader, desc=f"Validation: Epoch {epoch + 1}/{NUM_EPOCHS}", total=NUM_IMAGES_EVAL/BATCH_SIZE):
        model.eval()
        images, labels = images_labels['image'].to(device), images_labels['label'].to(device)
        adv_images = adv_model.pgd_attack(images, labels).to(device) # produce adv_images on-the-fly for validation
        with torch.no_grad():
            outputs = model(adv_images)
            predictions = outputs.softmax(1).argmax(dim=1)
            total += labels.size(0)
            correct += torch.sum(predictions == labels).item()

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {running_loss/(2*NUM_IMAGES_TRAIN):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")