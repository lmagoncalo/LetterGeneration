import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, Dataset


def load_dataset():
    rawdata = np.load('character_font.npz')
    images = rawdata['images']
    labels = rawdata['labels']
    images = torch.FloatTensor(images).unsqueeze(1).repeat(1, 3, 1, 1)
    labels = torch.Tensor(labels)

    images /= 255
    # print(images.shape, labels.shape)

    dataset = TensorDataset(images, labels)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    print("Train size:", train_size, "Test size:", test_size)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

    return train_loader, test_loader


class SketchDataset(Dataset):
    def __init__(
            self,
            image_paths,
            category,  # len(category)=10
            mode="train",
            Nmax=96,
    ):
        super().__init__()

        self.sketches = None
        self.sketches_normed = None
        self.max_sketches_len = 0
        self.path = image_paths
        self.category = category
        self.mode = mode
        self.Nmax = Nmax

        tmp_sketches = []
        tmp_label = []

        for i, c in enumerate(self.category):
            dataset = np.load(os.path.join(self.path, c), encoding='latin1', allow_pickle=True)
            tmp_sketches.append(dataset[self.mode])
            tmp_label.append([i] * len(dataset[self.mode]))
            print(f"dataset: {c} added.")

        data_sketches = np.concatenate(tmp_sketches)
        data_sketches_label = np.concatenate(tmp_label)
        data_sketches, data_sketches_label = self.purify(data_sketches, data_sketches_label)
        self.sketches = data_sketches.copy()
        self.sketches_label = data_sketches_label.copy()
        self.sketches_normed = self.normalize(data_sketches)

        # print(f"length of trainset(normed): {len(self.sketches_normed)}")
        self.len_dataset = len(self.sketches_normed)

        # self.Nmax = self.max_size(data_sketches)  # max size of a sk  etch.

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        sketch = np.zeros((self.Nmax, 3))
        len_seq = self.sketches_normed[idx].shape[0]
        sketch[:len_seq, :] = self.sketches_normed[idx]

        labels = None
        if self.sketches_label is not None:
            labels = np.array(self.sketches_label[idx], dtype=np.int64)

        return sketch, labels

    def max_size(self, sketches):
        sizes = [len(sketch) for sketch in sketches]
        return max(sizes)

    def purify(self, sketches, labels):
        data = []
        new_labels = []
        for i, sketch in enumerate(sketches):
            # if hp.max_seq_length >= sketch.shape[0] > hp.min_seq_length:  # remove small and too long sketches.
            if 96 >= sketch.shape[0] > 0:  # remove small and too long sketches.
                sketch = np.minimum(sketch, 1000)  # remove large gaps.
                sketch = np.maximum(sketch, -1000)
                sketch = np.array(sketch, dtype=np.float32)  # change it into float32
                data.append(sketch)
                new_labels.append(labels[i])
        return data, new_labels

    def calculate_normalizing_scale_factor(self, sketches):
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def normalize(self, sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(sketches)
        for sketch in sketches:
            sketch[:, 0:2] /= scale_factor
            data.append(sketch)
        return data


class FontDataset(Dataset):
    def __init__(
            self,
            fonts_path,
            Nmax=150,
    ):
        super().__init__()

        self.sketches = None
        self.sketches_normed = None
        self.max_sketches_len = 0
        self.path = fonts_path
        self.Nmax = Nmax

        dataset = np.load(self.path, encoding='latin1', allow_pickle=True)
        data_sketches = dataset["data"]

        data_labels = dataset["labels"]

        data_sketches, data_labels = self.purify(data_sketches, data_labels)
        self.sketches = data_sketches.copy()
        self.labels = data_labels.copy()
        self.sketches_normed = self.normalize(data_sketches)

        # print(f"length of trainset(normed): {len(self.sketches_normed)}")
        self.len_dataset = len(self.sketches_normed)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        sketch = np.zeros((self.Nmax, 3))
        len_seq = self.sketches_normed[idx].shape[0]
        sketch[:len_seq, :] = self.sketches_normed[idx]

        label = np.array(self.labels[idx], dtype=np.int64)

        return sketch, label

    def max_size(self, sketches):
        sizes = [len(sketch) for sketch in sketches]
        return max(sizes)

    def purify(self, sketches, labels):
        data = []
        new_labels = []
        for i, sketch in enumerate(sketches):
            # if hp.max_seq_length >= sketch.shape[0] > hp.min_seq_length:  # remove small and too long sketches.
            if self.Nmax >= sketch.shape[0] > 0:  # remove small and too long sketches.
                sketch = np.minimum(sketch, 1000)  # remove large gaps.
                sketch = np.maximum(sketch, -1000)
                sketch = np.array(sketch, dtype=np.float32)  # change it into float32
                data.append(sketch)
                new_labels.append(labels[i])
        return data, new_labels

    def calculate_normalizing_scale_factor(self, sketches):
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def normalize(self, sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(sketches)
        for sketch in sketches:
            x_max, y_max, _ = np.max(sketch, axis=0)
            sketch[:, 0:2] /= scale_factor
            data.append(sketch)

        return data
