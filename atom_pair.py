import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import pandas as pd
import pickle
from calc_property import calculate_property
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

atom_pair_index = pickle.load(open('new_atom_pair_vocab.pkl', 'rb') )

def get_vocab( smi ):
    m = get_geom_rdkit( smi , mode ='precise') #type(output) = mol
    if m is None:
        print(f'!!!!!!!!!!! Failed case for get_vocab {smi} {m}')
        return None
    dist_mat = Chem.Get3DDistanceMatrix(m)
    vocab_list = []
    for bond in m.GetBonds():
        i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i_atom, j_atom = m.GetAtomWithIdx(i).GetSymbol(), m.GetAtomWithIdx(j).GetSymbol()
        vocab_key = f'{i_atom}-{j_atom}'
        vocab_list.append(vocab_key)
    return vocab_list

def get_geom_rdkit( smi , max_try=5, mode='normal'):
    mol = Chem.AddHs( Chem.MolFromSmiles(smi) )
    if mol is None:
        return None
    num_try= 0
    while num_try < max_try:
        try:
            if mode == 'precise':
#                print('precise mode')
                mol = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.useSmallRingTorsions = True
                params.useExpTorsionAnglePrefs = True
                params.useBasicKnowledge = True
                params.pruneRmsThresh = 0.5
                params.maxAttempts = 1000
                params.randomSeed = 42
            
                conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=25, params=params)#max_confs=25
                if not conf_ids:
                    print('failed embedding')
                    raise RuntimeError(f"Embedding failed for {smi}")
                if not AllChem.MMFFHasAllMoleculeParams(mol):
                    print('failed mmff94 params')
                    raise RuntimeError(f"MMFF94s parameters unavailable for {smi}")
            
                props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
                best_conf_id = None
                min_energy = float('inf')
                for cid in conf_ids:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                    ff.Minimize(maxIts=2000)
                    energy = ff.CalcEnergy()
                    if energy < min_energy:
                        min_energy = energy
                        best_conf_id = cid
            
                if best_conf_id is None:
                    print('failed search best conf id')
                    raise RuntimeError(f"All conformers failed for {smi}")
            
                best_conf = mol.GetConformer(best_conf_id)
                best_conf_cp = Chem.Conformer(best_conf)
                mol.RemoveAllConformers()
                mol.AddConformer(best_conf_cp, assignId=True)
                return mol
            else:
#                print('normal model')
                AllChem.EmbedMolecule( mol ) 
                AllChem.MMFFOptimizeMolecule( mol )
                return mol

        except Exception as e:
            num_try+=1
            continue
    print(f'Failed to get geo max_try {max_try}: {smi}')

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
        #vocab_idx = atom_pair_index.get(f'{i_atom}-{j_atom}')
        vocab_idx = atom_pair_index.get(f'{i_atom}-{j_atom}', atom_pair_index['[UNK]']) #assign UNK when the atom-pair excluded in vocab
        vocab_list.append(vocab_idx)
        dist_list.append(dist)
    return torch.tensor(vocab_list).long(), torch.tensor(dist_list).float()

def safe_get_dist(args):
    property_mean, property_std = pickle.load(open('./normalize.pkl', 'rb') )
    idx, smi = args
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False, canonical=True)
        prop = ( calculate_property(smi) - property_mean) / property_std
        dist = get_dist(smi)
        return smiles, prop, dist
    except Exception as e:
        print(f'Failed {idx}: {e}')
        return None

if __name__ == '__main__':
    import time
    list_smi = [line.strip() for line in open('./data/1_Pretrain/pretrain_50m.txt')][1000000:1001000]
    st = time.time()
    with Pool(24) as p:
        results = list(tqdm(p.map(get_vocab, enumerate(list_smi) ), total=len(list_smi)))
#        results = list(tqdm(p.imap(get_vocab, list_smi), total=len(list_smi)))
#        results = list(tqdm(p.imap_unordered(get_vocab, list_smi, chunksize=10), total=len(list_smi)))
    print('end of get_vocab')
    et = time.time()
    results = [item for item in results if item is not None]
    print(f'Time for get geo:  {et-st:.2f}, {len(results)}')
    exit(-1)
    list_tmp_vocab = [item for item in results if None not in item]
    vocab_dict = Counter(list_tmp_vocab)
    print(vocab_dict)

    #for line in output:
        #print(line)
    df = pd.DataFrame(output, columns=['atom_pair', 'dist'])
    print(df)
    print(atom_pair_dict)
    vocab_idx = [ output[idx][0] for idx in range(len(output))]
    print(vocab_idx)
    vocab_idx = [ atom_pair_dict.get(output[idx][0]) for idx in range(len(output))]
    print(vocab_idx)
    df['idx'] = vocab_idx
    print(df)
    exit(-1)
