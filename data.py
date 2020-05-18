#from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import pdb
import os.path
from numpy.testing import assert_array_almost_equal

class TextData():
  def __init__(self, text_file, label_file, source_batch_size=64, target_batch_size=64, val_batch_size=4):
    all_text = np.load(text_file)
    self.source_text = all_text[0:92664, :]
    self.target_text = all_text[92664:, :]
    self.val_text = all_text[0:92664, :]
    all_label = np.load(label_file)
    self.label_source = all_label[0:92664, :]
    self.label_target = all_label[92664:, :]
    self.label_val = all_label[0:92664, :]
    self.scaler = StandardScaler().fit(all_text)
    self.source_id = 0
    self.target_id = 0
    self.val_id = 0
    self.source_size = self.source_text.shape[0]
    self.target_size = self.target_text.shape[0]
    self.val_size = self.val_text.shape[0]
    self.source_batch_size = source_batch_size
    self.target_batch_size = target_batch_size
    self.val_batch_size = val_batch_size
    self.source_list = random.sample(range(self.source_size), self.source_size)
    self.target_list = random.sample(range(self.target_size), self.target_size)
    self.val_list = random.sample(range(self.val_size), self.val_size)
    self.feature_dim = self.source_text.shape[1]
    
  def next_batch(self, train=True):
    data = []
    label = []
    if train:
      remaining = self.source_size - self.source_id
      start = self.source_id
      if remaining <= self.source_batch_size:
        for i in self.source_list[start:]:
          data.append(self.source_text[i, :])
          label.append(self.label_source[i, :])
          self.source_id += 1
        self.source_list = random.sample(range(self.source_size), self.source_size)
        self.source_id = 0
        for i in self.source_list[0:(self.source_batch_size-remaining)]:
          data.append(self.source_text[i, :])
          label.append(self.label_source[i, :])
          self.source_id += 1
      else:
        for i in self.source_list[start:start+self.source_batch_size]:
          data.append(self.source_text[i, :])
          label.append(self.label_source[i, :])
          self.source_id += 1
      remaining = self.target_size - self.target_id
      start = self.target_id
      if remaining <= self.target_batch_size:
        for i in self.target_list[start:]:
          data.append(self.target_text[i, :])
          # no target label
          #label.append(self.label_target[i, :])
          self.target_id += 1
        self.target_list = random.sample(range(self.target_size), self.target_size)
        self.target_id = 0
        for i in self.target_list[0:self.target_batch_size-remaining]:
          data.append(self.target_text[i, :])
          #label.append(self.label_target[i, :])
          self.target_id += 1
      else:
        for i in self.target_list[start:start+self.target_batch_size]:
          data.append(self.target_text[i, :])
          #label.append(self.label_target[i, :])
          self.target_id += 1
    else:
      remaining = self.val_size - self.val_id
      start = self.val_id
      if remaining <= self.val_batch_size:
        for i in self.val_list[start:]:
          data.append(self.val_text[i, :])
          label.append(self.label_val[i, :])
          self.val_id += 1
        self.val_list = random.sample(range(self.val_size), self.val_size)
        self.val_id = 0
        for i in self.val_list[0:self.val_batch_size-remaining]:
          data.append(self.val_text[i, :])
          label.append(self.label_val[i, :])
          self.val_id += 1
      else:
        for i in self.val_list[start:start+self.val_batch_size]:
          data.append(self.val_text[i, :])
          label.append(self.label_val[i, :])
          self.val_id += 1
    data = self.scaler.transform(np.vstack(data))
    label = np.aavstack(label)
    return torch.from_numpy(data).float(),torch.from_numpy(label).float()


def make_dataset(image_list, labels, domain_label = 1):
    x = np.array((labels!=None))
    if x.any():
      len_ = len(image_list)
      images = [(image_list[i].split()[0], labels[i], float(domain_label)) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1]), float(domain_label)) for val in image_list]
        
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
        return pil_loader(path)




class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, list1, list2=None, labels=None, domain = "train", transform=None, target_transform=None,
                 loader=default_loader):
        if(domain == "train"):
            imgs1 = make_dataset(list1, labels, 1)
            if(list2!= None):
                imgs2 = make_dataset(list2, labels, 0)
                imgs = imgs1 + imgs2
            else:
                imgs = imgs1
        else:
            imgs = make_dataset(list1, labels, 0)


        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target, domain = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, domain, index

    def __len__(self):
        return len(self.imgs)

class ImageList_coteaching(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs_noisy = make_dataset(image_list, labels)
        imgs_clean = make_dataset(image_list, None)
        if len(imgs_noisy) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs_noisy
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        train_labels, train_noisy_labels = list(), list()
        for i in range(len(self.imgs)):
          train_noisy_labels.append(imgs_noisy[i][1])
          train_labels.append(imgs_clean[i][1])
        self.noise_or_not = np.array(train_noisy_labels)==np.array(train_labels)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    #print (m)
    new_y = y.copy()
    # flipper = np.random.RandomState(random_state)
    flipper = np.random.RandomState() #by Rahul

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    #print (P)

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    #print (P)

    return y_train, actual_noise

def noisify(dataset='office', nb_classes=31, train_labels=None, noise_type='symmetric', noise_rate=0.4, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate





class ImageList_debug(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader, nb_classes= 31, noise_type='symmetric', noise_rate = 0.4, random_state=0):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.nb_classes = nb_classes
        self.train_labels = list()
        for i in range(len(self.imgs)):
          self.train_labels.append(imgs[i][1])
        
        self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
        self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=image_list, nb_classes=self.nb_classes, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
        
        self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
        _train_labels=[i[0] for i in self.train_labels]
        self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index][0], self.train_noisy_labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


class prepare_dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_noisy_labels = dataset.train_noisy_labels
        self.train_labels = dataset.train_labels
        self.noise_or_not = dataset.noise_or_not

        self.idx1   = torch.from_numpy(np.zeros(len(dataset))).byte()
        self.idx2   = torch.from_numpy(np.zeros(len(dataset))).byte()
        self.relbl1 = torch.from_numpy(np.zeros(len(dataset))).long()
        self.relbl2 = torch.from_numpy(np.zeros(len(dataset))).long()
    
    def __getitem__(self, index):
        
        img, target, index = self.dataset[index]
        clean_target = self.dataset.train_labels[index][0]
        idx1 = self.idx1[index]
        idx2 = self.idx2[index]
        relbl1 = self.relbl1[index]
        relbl2 = self.relbl2[index]

        return img, target, idx1, idx2, relbl1, relbl2, index, clean_target
    
    def __len__(self):
      return len(self.dataset)
    
    