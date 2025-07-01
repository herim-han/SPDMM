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
import ray
import time
RDLogger.DisableLog('rdApp.*')
#torch.multiprocessing.set_sharing_strategy('file_system')
#multiprocessing.set_start_method("spawn", force=True)


@ray.remote
def data_preprocess(args):
    property_mean, property_std = pickle.load(open('./normalize.pkl', 'rb') )
    property_mean = property_mean.detach().cpu().numpy().astype(np.float32)
    property_std  = property_std.detach().cpu().numpy().astype(np.float32)
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
    with open('./property_name.txt', 'r') as f:
        names = [n.strip() for n in f.readlines()][:53]
    
    descriptor_dict = OrderedDict()
    for n in names:
        if n == 'QED':
            descriptor_dict[n] = lambda x: Chem.QED.qed(x)
        else:
            descriptor_dict[n] = getattr(Descriptors, n)

    mol = Chem.MolFromSmiles(smiles)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        output.append(descriptor_dict[descriptor](mol))
    return output

def to_numpy(x):
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return x

def get_dist( smi ):
    atom_pair_index = pickle.load(open('new_atom_pair_vocab.pkl', 'rb') )
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
    ray.init()
    st= time.time()
    list_smiles = [l.strip() for l in open('./data/1_Pretrain/pretrain_50m.txt').readlines()][:10000]
    futures = [data_preprocess.remote((idx,smi)) for idx, smi in enumerate(list_smiles) ]
    pickle.dump(futures, open('tmp_ray_SMILESDataset.pkl', 'wb'))
    st = time.time()
    results = [ray.get(f) for f in tqdm(futures, desc="transform ray object")]
    filterted_results = [r for r in results if r is not None and all(x is not None for x in r)]
    pickle.dump(filtered_results, open('ray_SMILESDataset.pkl', 'wb'))
    et = time.time()
    print('time for ray', et-st)
    exit(-1)


    futures = [data_preprocess.remote((idx, smi)) for idx, smi in enumerate(list_smiles)]
     
    results = []
    completed = set()
    with tqdm(total=len(futures)) as pbar:
        while len(completed) < len(futures):
            done, futures = ray.wait(futures, num_returns=1, timeout=1.0)
    
            for obj_ref in done:
                result = ray.get(obj_ref)
    
                # filtering
                if result is not None and all(x is not None for x in result):
                    results.append(result)
    
                completed.add(obj_ref)
                pbar.update(1)
    et= time.time() 
    print(et-st)
    # 저장
    with open("./ray_SMILESDataset.pkl", "wb") as f:
        pickle.dump(results, f)
    exit(-1)
    list_smiles = [l.strip() for l in open('./data/1_Pretrain/pretrain_50m.txt').readlines()][:10000]
    futures = []
    for idx, smi in tqdm(enumerate(list_smiles), total=len(list_smiles)):
        futures.append(data_preprocess.remote((idx, smi)))
    results = ray.get(futures)
    filtered_results = [item for item in results
                        if item is not None and all(x is not None for x in item) ]
    pickle.dump(filtered_results, open(f'./ray_SMILESDataset.pkl', 'wb'))
    exit(-1)
    completed = set()
    with tqdm(total=len(futures), desc="Processing results") as pbar:
        while len(completed) < len(futures):
            done, _ = ray.wait(futures, num_returns=1, timeout=0.2)
            for obj in done:
                if obj not in completed:
                    completed.add(obj)
                    pbar.update(1)
            time.sleep(0.01)

    with Pool(8) as p:
#        results = list(tqdm(p.map(data_preprocess, enumerate(list_smiles) ), total=len(list_smiles)) )
#        results = list(tqdm(p.imap(data_preprocess, enumerate(list_smiles) ), total=len(list_smiles)) )
        results = list(tqdm(p.imap_unordered(data_preprocess, enumerate(list_smiles), chunksize=1), total=len(list_smiles)))
    filtered_results = [item for item in results
                        if item is not none and all(x is not none for x in item) ]
#    with open('map_SMILESDataset.pkl', 'wb') as f:
    with open('imap_unorder_SMILESDataset.pkl', 'wb') as f:
        pickle.dump(filtered_results, f)

