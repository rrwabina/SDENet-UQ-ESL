import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.slices = []

        for i, d in enumerate(data):
            for j in range(d['image'].shape[0]):
                self.slices.append((i, j))
    

    def __getitem__(self, idx):
        slc = self.slices[idx]
        sample = dict()
        sample['id'] = idx
        vol_id = slc[0]
        slice_num = slc[1]

        sample['image'] = torch.tensor(self.data[vol_id]['image'][slice_num][None, :])
        sample['seg'] = torch.tensor(self.data[vol_id]['seg'][slice_num][None, :])

        return sample

    def __len__(self):
        return len(self.slices)