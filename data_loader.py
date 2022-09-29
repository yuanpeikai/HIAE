from helper import *

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:	The triples used for training the model
    params:		Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params

        self.entities = np.arange(self.p.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, quals, label, N = torch.LongTensor(ele['triple']), torch.LongTensor(ele['quals']), np.int32(
            ele['label']), torch.tensor(ele['N'])
        trp_label = self.get_label(label)

        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)

        return triple, trp_label, quals, N

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        quals = torch.stack([_[2] for _ in data], dim=0)
        N = torch.stack([_[3] for _ in data], dim=0)
        return triple, trp_label, quals, N

    def get_label(self, label):

        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0

        return torch.FloatTensor(y)


class TestDataset(Dataset):
    """
    Evaluation Dataset class.

    Parameters
    ----------
    triples:	The triples used for evaluating the model
    params:		Parameters for the experiments

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, quals, label, N = torch.LongTensor(ele['triple']), torch.LongTensor(ele['quals']), np.int32(
            ele['label']), torch.tensor(ele['N'])
        label = self.get_label(label)
        return triple, label, quals, N

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        quals = torch.stack([_[2] for _ in data], dim=0)
        N = torch.stack([_[3] for _ in data], dim=0)
        return triple, label, quals, N

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)
