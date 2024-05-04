from torchvision import datasets
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image,make_grid
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])
train_set = datasets.MNIST("/home/allenyljiang/Desktop/d2l-pytorch/data", train=True, download=False,transform=pipeline)
test_set = datasets.MNIST("/home/allenyljiang/Desktop/d2l-pytorch/data", train=False, download=False,transform=pipeline)
train_label_indices = [2]
test_label_indices = [3]
subset_trainset = torch.utils.data.Subset(train_set,
                                          [i for i in range(len(train_set)) if train_set[i][1] in train_label_indices])
train_dataloader = DataLoader(subset_trainset, batch_size=128, num_workers=8, shuffle=True)
train_features,train_labels = next(iter(train_dataloader))
print(f"feature batch shape:{train_features.size()}") # [8, 1, 28, 28]
print(f"labels batch shape:{train_labels.size()}")
print(f"train dataset {len(train_dataloader)} ") # 47*128

# images = make_grid(train_features,nrow=1)
# save_image(images,'train_batch_example.png')