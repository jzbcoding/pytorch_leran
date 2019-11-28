#%%
import os
import torch
import torchvision
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")
plt.ion()

#%%
landmarks_frame = pd.read_csv("pytorch_learn//data//faces//faces//face_landmarks.csv")
n=65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1,2)

print('image name:{}'.format(img_name))
print('landmarks shape:{}'.format(landmarks.shape))
print('first 4 landmarks:{}'.format(landmarks[:4]))


#%%
def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:,0], landmarks[:,1:], s=10, marker='.', c='r')
    plt.pause(0.001)
plt.figure()
show_landmarks(io.imread(os.path.join('pytorch_learn//data//faces//faces//', img_name)), landmarks)
plt.show()


#%% [markdown]
# 数据集类
#%%
class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame=pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1,2)
        sample = {'image':image, 'landmarks':landmarks}
        if self.transform:
            sample = self.transform(sample)

        return sample

face_dataset = FaceLandmarksDataset(csv_file="pytorch_learn//data//faces//faces//face_landmarks.csv", 
                                    root_dir="pytorch_learn//data//faces//faces//")
# fig = plt.figure()
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#     print(i, sample['image'].shape, sample['landmarks'].shape)
#     ax = plt.subplot(1,4,i+1)
#     plt.tight_layout()
#     ax.set_title('sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample)
#     if i==3:
#         plt.show()
#         break
# plt.close('all')
#%% [markdown]
# 通过上面的例子我们会发现图片并不是同样的尺寸。绝大多数神经网络都假定图片的尺寸相同。
# 因此我们需要做一些预处理。让我们创建三个转换:

# - Rescale: 缩放图片
# - RandomCrop: 对图片进行随机裁剪。这是一种数据增强操作
# - ToTensor: 把 numpy 格式图片转为 torch 格式图片 (我们需要交换坐标轴).
'''
我们会把它们写成可调用的类的形式而不是简单的函数，这样就不需要每次调用时传递一遍参数。
我们只需要实现 __call__ 方法，必要的时候实现 __init__ 方法。我们可以这样调用这些转换:
'''
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if(h > w):
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        # 关键点坐标也要跟着变化
        landmarks = landmarks * [new_w/w, new_h/h]
        return {'image':img, 'landmarks':landmarks}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        image = image[top:top + new_h, 
                      left:left + new_w]
        landmarks = landmarks - [left, top]
        return {'image':image, 'landmarks':landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2,0,1))
        return {'image':torch.from_numpy(image),
                'landmarks':torch.from_numpy(landmarks)}


#%%
scale = Rescale(246)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), RandomCrop(224)])

fig = plt.figure()
sample = face_dataset[65]
for im, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)
    ax = plt.subplot(1,3,im+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)
plt.show()


#%% [markdown]
# - 迭代数据集 
transformed_dataset = FaceLandmarksDataset(csv_file="pytorch_learn//data//faces//faces//face_landmarks.csv", 
                                           root_dir="pytorch_learn//data//faces//faces//",
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

#%%
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break


#%%
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=0)

#%%
