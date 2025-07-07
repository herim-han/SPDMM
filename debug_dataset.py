from torch.utils.data import Dataset
import random
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import torch
import pandas as pd
import pickle
from collections import Counter, OrderedDict
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
RDLogger.DisableLog('rdApp.*')
torch.multiprocessing.set_sharing_strategy('file_system')
multiprocessing.set_start_method("spawn", force=True)

with open('./property_name.txt', 'r') as f:
    names = [n.strip() for n in f.readlines()][:53]

descriptor_dict = OrderedDict()
for n in names:
    if n == 'QED':
        descriptor_dict[n] = lambda x: Chem.QED.qed(x)
    else:
        descriptor_dict[n] = getattr(Descriptors, n)

atom_pair_index = pickle.load(open('new_atom_pair_vocab.pkl', 'rb') )
property_mean, property_std = pickle.load(open('./normalize.pkl', 'rb') )
property_mean = property_mean.detach().cpu().numpy().astype(np.float32)
property_std  = property_std.detach().cpu().numpy().astype(np.float32)

def data_preprocess(args):
    idx, smiles = args
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False, canonical=True)
        prop = calculate_property(smiles)
        properties = (prop - property_mean) / property_std #numpy, numpy, numpy
        atom_set, dist = get_dist(smiles) or (None, None)
#        print('55555', type(atom_set), type(dist))
        atom_set = to_numpy(atom_set)
        dist = to_numpy(dist)
        return properties, smiles, atom_set, dist
    except Exception as e:
        print(f"Failed processing {idx}: {e}")
        return None
        
def calculate_property(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        # print(descriptor)
        output.append(descriptor_dict[descriptor](mol))
#    return torch.tensor(output, dtype=torch.float)
    return output

def to_numpy(x):
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return x

def get_dist( smi ):
    m = get_geom_rdkit( smi ) #type(output) = mol
    if m is None:
        return None
    dist_mat = Chem.Get3DDistanceMatrix(m)
    vocab_list = []
    dist_list  = []
    for bond in m.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i_atom, j_atom = m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(j).GetSymbol()
        dist = dist_mat[i,j]
        vocab_idx = atom_pair_index.get(f'{i_atom}-{j_atom}', atom_pair_index['[UNK]']) #assign UNK when the atom-pair excluded in vocab
        vocab_list.append(vocab_idx)
        dist_list.append(dist)
    #return torch.tensor(vocab_list).long(), torch.tensor(dist_list).float()
    return vocab_list, dist_list
            
def get_geom_rdkit( smi , max_try=5, mode='normal'):
    mol = Chem.AddHs( Chem.MolFromSmiles(smi) )
    if mol is None:
        return None
    num_try= 0
    while num_try < max_try:
        try:
#            print('normal model')
            AllChem.EmbedMolecule( mol ) 
            AllChem.MMFFOptimizeMolecule( mol )
            return mol

        except Exception as e:
            num_try+=1
            continue
    print(f'Failed to get geo max_try {max_try}: {smi}')

if __name__=='__main__':
    import os
    import time
#    list_smiles = [l.strip() for l in open('./data/1_Pretrain/pretrain_50m.txt').readlines()][:10000]
    list_name = sorted(os.listdir('./pubchem_chunk'))[:3]
    print(list_name)
    
    for name in list_name:
        print(name)
        list_smiles = [line.strip() for line in open(f'./pubchem_chunk/{name}').readlines()]
        st = time.time()
        try:
            with Pool(24) as p:
        #        results = list(tqdm(p.map(data_preprocess, enumerate(list_smiles) ), total=len(list_smiles)) )
        #        results = list(tqdm(p.imap(data_preprocess, enumerate(list_smiles) ), total=len(list_smiles)) )
                results = list(tqdm(p.imap_unordered(data_preprocess, enumerate(list_smiles), chunksize=1), total=len(list_smiles)))
            et = time.time()
            print(f'Time for {name} file: {et-st:.3f} ')
            filtered_results = [item for item in results
                                if item is not None and all(x is not None for x in item) ]
        #    with open('map_SMILESDataset.pkl', 'wb') as f:
            with open(f'./Dataset/{name}_SMILESDataset.pkl', 'wb') as f:
                pickle.dump(filtered_results, f)
        except Exception as e:
            print(f'Error occur : {name}, {e}')
