from torch.utils.data import Dataset
import torch
import random
import pandas as pd
from rdkit import Chem
import pickle
from rdkit import RDLogger
from calc_property import calculate_property
from pysmilesutils.augment import MolAugmenter
from atom_pair import get_dist
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from multiprocessing import Pool
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')
torch.multiprocessing.set_sharing_strategy('file_system')

class SMILESDataset_pretrain(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        self.smiles = [l.strip() for l in open(data_path).readlines()]

        with Pool(24) as p:
            results = list(tqdm(p.imap(data_preprocess, enumerate(self.smiles) ), total=len(self.smiles)) )
#        self.data = [item for item in results if None not in item]
        self.data = [item for item in results if all(x is not None for x in item)]

        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def data_preprocess(args):
    idx, smiles = args
    property_mean, property_std = pickle.load(open('./normalize.pkl', 'rb') )
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False, canonical=True)
        properties = (calculate_property(smiles) - property_mean) / property_std
        atom_set, dist = get_dist(smiles) or (None, None)
        return properties, '[CLS]' + smiles, atom_set, dist

    except Exception as e:
        print(f"Failed processing {idx} {smiles}: {e}")
        return None
            
def collate_fn(batch):
    properties, smiles, atom_pair, dist = zip(*batch)
    properties = torch.stack(properties)
    atom_pair = pad_sequence(atom_pair, batch_first=True, padding_value=0)
    dist = pad_sequence(dist, batch_first=True, padding_value=0)
    return properties, smiles, atom_pair, dist

class SMILESDataset_BACEC(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['mol']), isomericSmiles=False, canonical=True)
        value = int(self.data[index]['Class'])

        return '[CLS]' + smiles, value


class SMILESDataset_BACER(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(6.420878294545455)
        self.value_std = torch.tensor(1.345219669175284)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = torch.tensor(self.data[index]['target'].item())
        return '[CLS]' + smiles, value


class SMILESDataset_LIPO(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(2.162904761904762)
        self.value_std = torch.tensor(1.210992810122257)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['exp'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_Clearance(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(51.503692077727955)
        self.value_std = torch.tensor(53.50834365711207)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['target'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_BBBP(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data)) if Chem.MolFromSmiles(data.iloc[i]['smiles'])]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        label = int(self.data[index]['p_np'])

        return '[CLS]' + smiles, label


class SMILESDataset_ESOL(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(-2.8668758314855878)
        self.value_std = torch.tensor(2.066724108076815)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['ESOL predicted log solubility in mols per litre'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_Freesolv(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        self.value_mean = torch.tensor(-3.2594736842105267)
        self.value_std = torch.tensor(3.2775297233608893)

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = (self.data[index]['target'] - self.value_mean) / self.value_std

        return '[CLS]' + smiles, value


class SMILESDataset_Clintox(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]
        self.n_output = 2

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = torch.tensor([float(self.data[index]['FDA_APPROVED']), float(self.data[index]['CT_TOX'])])

        return '[CLS]' + smiles, value


class SMILESDataset_SIDER(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]
        self.n_output = 27

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False, canonical=True, kekuleSmiles=False)
        value = self.data[index].values.tolist()[1:]
        value = torch.tensor([i.item() for i in value])
        return '[CLS]' + smiles, value


class SMILESDataset_DILI(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['Smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['Liver'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_USPTO(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False, aug=False):
        self.is_aug = aug
        self.aug = MolAugmenter()
        with open(data_path, 'r') as f:
            lines = f.readlines()
        self.data = [line.strip() for line in lines]

        if shuffle:
            random.shuffle(self.data)
        if data_length:
            self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rs, ps = self.data[index].split('\t')
        if self.is_aug and random.random() > 0.5:
            r_mol = self.aug([Chem.MolFromSmiles(rs[:])])[0]
            rs = Chem.MolToSmiles(r_mol, canonical=False, isomericSmiles=False)
            p_mol = self.aug([Chem.MolFromSmiles(ps[:])])[0]
            ps = Chem.MolToSmiles(p_mol, canonical=False, isomericSmiles=False)
        return '[CLS]' + rs, '[CLS]' + ps


class SMILESDataset_USPTO_reverse(Dataset):
    def __init__(self, data_length=None, shuffle=False, mode=None, aug=False):
        with open('./data/6_RXNprediction/USPTO-50k/uspto_50.pickle', 'rb') as f:
            data = pickle.load(f)
        data = [data.iloc[i] for i in range(len(data))]
        self.data = [d for d in data if d['set'] == mode]
        self.is_aug = aug
        self.aug = MolAugmenter()

        if shuffle:
            random.shuffle(self.data)
        if data_length:
            self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        # r_type = d['reaction_type']
        p_mol = d['products_mol']
        r_mol = d['reactants_mol']
        do_aug = self.is_aug and random.random() > 0.5
        if do_aug:
            p_mol = self.aug([p_mol])[0]
            r_mol = self.aug([r_mol])[0]
        return '[CLS]' + Chem.MolToSmiles(p_mol, canonical=not do_aug, isomericSmiles=False), \
               '[CLS]' + Chem.MolToSmiles(r_mol, canonical=not do_aug, isomericSmiles=False)
