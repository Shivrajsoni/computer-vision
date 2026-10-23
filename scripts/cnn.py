# Fine-tune a pretrained ResNet-18 on a small dataset (Oxford Flowers or CIFAR-100) with PyTorch
# Log accuracy and a confusion matrix.

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, models , transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

norm = transforms.Normalize([0.485,0.456,0.406] , [0.229,0.224,0.225])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    norm,
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    norm,
])

train_ds = datasets.Flowers102("data",split="train",transform=train_tf,download=True)
val_ds = datasets.Flowers102("data",split="val",transform=eval_tf,download=True)
test_ds = datasets.Flowers102("data",split="test",transform=eval_tf,download=True)


train_dl = DataLoader(train_ds,batch_size=32,shuffle=True)
val_dl = DataLoader(val_ds,batch_size=64)
test_dl = DataLoader(test_ds, batch_size=64)

num_classes = 102
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features,num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

def run_epoch(loader , optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct , n = 0.0 ,0,0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x,y = x.to(device) , y.to(device)
            out = model(x)
            loss = criterion(out , y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()* x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def fit(optimizer, epochs, scheduler=None):
    best = 0.0
    for ep in range(epochs):
        tl, ta = run_epoch(train_dl, optimizer)
        vl, va = run_epoch(val_dl)
        if scheduler:
            scheduler.step()
        print(f"ep {ep+1}/{epochs} train {tl:.3f}/{ta:.3f} val {vl:.3f}/{va:.3f}")
        if va > best:
            best = va
            torch.save(model.state_dict(), "outputs/best.pt")


# stage 1: head only
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True
fit(torch.optim.Adam(model.fc.parameters(), lr=1e-3), epochs=5)

# stage 2: unfreeze all, low LR for backbone
for p in model.parameters():
    p.requires_grad = True
opt = torch.optim.Adam([
    {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")], "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
fit(opt, epochs=15, scheduler=sched)

# test + confusion matrix
model.load_state_dict(torch.load("outputs/best.pt"))
model.eval()
preds, labels = [], []
with torch.no_grad():
    for x, y in test_dl:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()

acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
print(f"test acc: {acc:.4f}")

cm = confusion_matrix(labels, preds, normalize="true")
ConfusionMatrixDisplay(cm).plot(include_values=False, cmap="Blues")
plt.savefig("outputs/confusion_matrix.png", dpi=150)
