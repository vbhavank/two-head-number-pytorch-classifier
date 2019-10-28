import torch.nn as nn
import os
from os.path import join
import json
import copy
import torch
from PIL import Image
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import time
import torch.optim as optim
from torchvision import transforms


__author__ = 'bxv7657@rit.edu'

class MLBDataset(Dataset):
    """Data loader to train multi task network
    returns a '0' for first digit for number between 0-9 """

    def __init__(self,
                 dirpath,
                 tag,
                 img_w, img_h,
                 transforms,
                 batch_size,
                 max_text_len=2):
        """
         :param: dirpath: Path to the folder with images and annotations.
         :param: tag: 'train' , 'val': Flag to route train/validation data sample return.
         :param: img_w: Return Image width.
         :param: img_h: Return Image height.
         :param: max_text_len: Range of labels, default 0-99(2).
         :return: [PIL.Image, int, int]
         """

        self.img_h = img_h
        self.img_w = img_w
        self.tag = tag
        self.transform = transforms
        self.max_text_len = max_text_len
        self.img_dirpath = join(dirpath, 'img')
        self.ann_dirpath = join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(self.img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext in ['.png', '.jpg']:
                img_filepath = join(self.img_dirpath, filename)
                json_filepath = join(self.ann_dirpath, name + '.json')
                ann = json.load(open(json_filepath, 'r'))
                description = ann['description']
                if len(description) != max_text_len:
                    new_description = ['0', description] #Add label 0 for first branch if number is in range 0-9
                    description = new_description
                tags = ann['tags'][0]
                if not (self.tag == tags):
                    continue
                self.samples.append([img_filepath, description])
        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.batch_size = batch_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img_filepath, description = self.samples[idx]
        img = Image.open(img_filepath).convert('L')
        img = img.resize((self.img_h, self.img_w))
        return self.transform(img), int(description[0]), int(description[1])

    def get_counter(self):
        """
        Provides summary of the dataset.
        """

        # path to the directory containing data
        dirname = os.path.basename(self.img_dirpath)
        # path to the annotation data directory
        ann_dirpath = join(self.img_dirpath, 'ann')
        digits = ''
        lens = []
        for filename in os.listdir(ann_dirpath):
            json_filepath = join(ann_dirpath, filename)
            ann = json.load(open(json_filepath, 'r'))  # a dictionary with metadata for the image
            tags = ann['tags'][0]  # whether this image is part of the validation or train set
            if self.tag == tags:
                description = ann['description']  # the description is the number present in image
                lens.append(len(description))
                digits += description  # number of digits in the image text
        print('The maximum no. of digits in the dataset is: "%s"' % dirname, max(Counter(lens).keys()))
        return Counter(digits)  # returns the length of digits in the image


class Net(nn.Module):
    """Network architecture definition"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(40)
        self.conv2 = nn.Conv2d(40, 80, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(80, 160, kernel_size=5)
        self.avg_p = nn.AvgPool2d(kernel_size=4)
        self.fc1 = nn.Linear(640, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d((self.conv2_drop(self.conv2(x))), 2))
        x = F.relu(self.conv3(x))
        x = self.avg_p(x)
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p= 0.2,training=True)
        out = self.fc2(x)
        out1 = self.fc3(out)
        out2 = self.fc4(out)
        return F.log_softmax(out1,dim=1), F.log_softmax(out2,dim=1)


class JointLoss(nn.Module):
    """
    Loss function
    Cross-entropy loss used for joint-training of both branches
    """

    def __init__(self):
        super(JointLoss, self).__init__()

    def forward(self, output, labels, size_average=True):
        losses = F.cross_entropy(output, labels)
        return losses.mean() if size_average else losses.sum()

def train_model(model, criterion, optimizer, num_epochs):
    """
    Function to train multi task model.
    """

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    e_losses = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        losses = []
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_corrects_1 = 0
            running_corrects_2 = 0
            for inputs, labels1, labels2 in data_lo[phase]:
                labels1 = labels1.cuda()
                labels2 = labels2.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output_1, output_2 = model(inputs.cuda())
                    _, preds_1 = torch.max(output_1, 1)
                    _, preds_2 = torch.max(output_2, 1)
                    loss_1 = criterion(output_1, labels1)
                    loss_2 = criterion(output_2, labels2)
                    loss = loss_1 + loss_2
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.data.cpu().numpy())
                running_loss += loss.item() * inputs.size(0)
                running_corrects_1 += torch.sum(preds_1 == labels1 )
                running_corrects_2 += torch.sum(preds_2 == labels2 )
                running_corrects= (running_corrects_1 + running_corrects_2)//2
            epoch_loss = running_loss / (dataset_sizes[phase] )
            epoch_accuracy = running_corrects.double() /  (dataset_sizes[phase])
            e_losses.append(epoch_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_accuracy))
            if phase == 'val' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
        print("\n")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, best_acc, e_losses

def test(model,test_loader):
    """
    Function to test model
    Returns accuracy on validation data samples.
    """

    accuracy = 0
    model.eval()
    for inputs, labels1, labels2 in test_loader:
        inputs = inputs.cuda()
        labels1 = labels1.cuda()
        labels2 = labels2.cuda()
        output_1, output_2 = model.forward(inputs)
        _, preds_1 = torch.max(output_1, 1)
        _, preds_2 = torch.max(output_2, 1)
        equality1 = (labels1.data == preds_1)
        equality2 = (labels2.data == preds_2)
        accuracy += ((equality1 + equality2)//2).type(torch.FloatTensor).mean()
    return accuracy/len(test_loader)*inputs.size(0)

train_batch_size = 4
val_batch_size = 1
learning_rt = 0.0001
number_epochs = 1000
data_dir = "./screencap_v2017/" #Change to appropriate dataset location
image_transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'val': transforms.Compose([ transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
}
MLB_obj_train = MLBDataset(data_dir,'train', 60, 60, image_transform['train'],train_batch_size)
train_loader = torch.utils.data.DataLoader(MLB_obj_train,batch_size=train_batch_size, shuffle=True)
MLB_obj_test = MLBDataset(data_dir,'val', 60, 60, image_transform['val'],val_batch_size)
test_loader = torch.utils.data.DataLoader(MLB_obj_test,batch_size=val_batch_size, shuffle=True)
print("Number of training examples loaded: {}".format(len(train_loader)*train_batch_size))
print("Number of validation examples loaded: {}".format(len(test_loader)*val_batch_size))
dataset_sizes={
    'train':len(train_loader)*train_batch_size,
    'val':len(test_loader)*val_batch_size
}
data_lo = {
    'train': train_loader,
    'val': test_loader
}
joint_model = Net().cuda()
criterion = JointLoss()
optimizer_ft = optim.Adam(joint_model.parameters(), lr=learning_rt, betas=(0.9, 0.999))
model_ft, best_acc, e_losses = train_model(joint_model, criterion, optimizer_ft, num_epochs=number_epochs)
print('Finished Training')
torch.save(model_ft.state_dict(), ("./multitask_sunday_{:2f}.pth".format(int(best_acc * 100))))
plt.plot(e_losses)
plt.show()
epoch_acc = test(model_ft,test_loader)
print(' Test Acc: {:.4f}'.format(epoch_acc))




