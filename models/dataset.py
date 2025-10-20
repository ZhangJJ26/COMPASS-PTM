import torch
from torch.utils.data import Dataset, ConcatDataset
import os
import pandas as pd
import time
import numpy as np
import json

class PTMDataset_(Dataset):
    def __init__(self, sequences, labels, tokenizer, seq_lens, max_length=50, kinases=None):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.seq_lens = seq_lens
        self.max_length = max_length
        self.kinases = kinases
        if labels == None:
            self.with_labels = False
        else:
            self.with_labels = True
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        if self.with_labels:
            label = self.labels[idx]
        else:
            ptm_types_num = 8  # number of PTM types including 'others'
            label = np.zeros((self.max_length-20, ptm_types_num))  # dummy label for inference
        seq_len = self.seq_lens[idx]
        
        # pad sequence to max_length
        sequence = sequence.ljust(self.max_length, 'X')
        # Tokenize using Hugging Face tokenizer
        tokens = self.tokenizer(
            sequence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length + 2,  # +2 for special tokens
            return_tensors='pt'
        )
        
        # print("tokens:", tokens)
        input_ids = tokens['input_ids'].squeeze(0)
        # print("input_ids:", input_ids)
        attention_mask = tokens['attention_mask'].squeeze(0)
        # print("attention_mask:", attention_mask)
        if self.kinases is not None:
            kinase = self.kinases[idx]
            return input_ids, torch.tensor(label, dtype=torch.float), attention_mask, seq_len, kinase
        else:
            return input_ids, torch.tensor(label, dtype=torch.float), attention_mask, seq_len, sequence
        

class PTMDataset_finetune_binary(Dataset):
    def __init__(self, sequences, labels, tokenizer, seq_lens, max_length=21, kinases=None):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.seq_lens = seq_lens
        self.max_length = max_length
        self.kinases = kinases

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        seq_len = self.seq_lens[idx]
        
        # pad sequence to max_length
        sequence = sequence.ljust(self.max_length, 'X')
        # Tokenize using Hugging Face tokenizer
        tokens = self.tokenizer(
            sequence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length + 2,  # +2 for special tokens
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        kinase = self.kinases[idx]
        return input_ids, torch.tensor(label, dtype=torch.float), attention_mask, seq_len, kinase, sequence


def load_data_(csv_file, tokenizer, max_len=70):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Extract sequences and labels
    peps = df['Peptide_sequence'].tolist()
    seqs = df['Sequence'].tolist()
    seq_lens = [len(seq) for seq in peps]
    max_pep_len = max(seq_lens)
    print(f"Max peptide length: {max_pep_len}")
    _sites = df['Sites'].tolist()
    _ptms = df['PTMs'].tolist()
    _starts = df['Start'].tolist()
    _uniprot_ids = df['Uniprot_ID'].tolist()
    labels = []

    ptm_list = [
            'ADP-ribosylation', 'Acetylation', 'Glutathionylation', 'Malonylation',
            'Methylation', 'Phosphorylation', 'S-nitrosylation', 'S-palmitoylation',
            'Succinylation', 'Sulfoxidation', 'Sumoylation', 'Ubiquitination',
            'O-linked Glycosylation', 'Lactylation', 'Hydroxylation', 'Crotonylation',
            'Glutarylation', 'Lactoylation', 'Citrullination', 'Neddylation', 'Formylation',
            'N-linked Glycosylation', 'Amidation', 'Dephosphorylation', 'Myristoylation',
            'Oxidation', 'Pyrrolidone carboxylic acid', 'Sulfation', 'GPI-anchor',
            'Gamma-carboxyglutamic acid', 'C-linked Glycosylation'
        ]

    high_freq_labels = [1, 4, 5, 8, 11, 12, 21]
    high_freq_ptm_list = [ptm_list[i] for i in high_freq_labels]
    ptm2idx = {ptm: i for i, ptm in enumerate(high_freq_ptm_list)}
    for sites, ptms, start in zip(_sites, _ptms, _starts):
        site_list = [int(s) for s in sites[1:-1].replace('\'', '').replace('\"', '').split(', ')]
        site_labels = np.zeros(max_pep_len-20)
        start = int(start)
        site_labels[np.array(site_list, dtype=int) - start] = 1
        
        __ptm_list = ptms[1:-1].replace('\'', '').replace('\"', '').split(', ')
        ptm_labels_dict = {site: np.zeros(len(ptm2idx)+1) for site in range(max_pep_len-20)}
        for site, ptm in zip(site_list, __ptm_list):
            if ';' in ptm:
                ptm = ptm.split(';')
                for p in ptm:
                    if p in high_freq_ptm_list:
                        ptm_labels_dict[site - start][ptm2idx[p]] = 1
                    else:
                        ptm_labels_dict[site - start][-1] = 1
            else:
                if ptm in high_freq_ptm_list:
                    ptm_labels_dict[site - start][ptm2idx[ptm]] = 1
                else:
                    ptm_labels_dict[site - start][-1] = 1
                
        label = [ptm_labels_dict[site] for site in range(max_pep_len-20)]
        labels.append(label)
    
    labels = np.array(labels)
    labels = torch.tensor(labels, dtype=torch.float32)
    seq_lens = [length - 20 for length in seq_lens]
    seq_lens = np.array(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.float32)

    print(f"Total sequences: {len(peps)}")
    print(f"Label shape: {labels.shape}")
    
    return PTMDataset_(peps, labels, tokenizer, seq_lens, max_length=max_len)


def load_data_finetune_binary(csv_file, tokenizer, max_len=50):
    # Read CSV file
    df = pd.read_csv(csv_file)
    peps = df['peptide'].tolist()
    kinase_names = df['kinase_name'].tolist()
    seq_lens = [len(seq) for seq in peps]
    max_pep_len = max(seq_lens)
    print(f"Max peptide length: {max_pep_len}")
    labels = df['label'].tolist()
    
    kinases = []

    with open('../data/kinases_dict_sagephos.json', 'r') as f:
        kinase_dict = json.load(f)
    
    kinase2idx = {v: k for k, v in kinase_dict.items()}
    
    for kinase_name in kinase_names:
        kinase_id = int(kinase2idx[kinase_name.upper()])
        kinases.append(kinase_id)

    labels = np.array(labels)
    labels = torch.tensor(labels, dtype=torch.float32)
    seq_lens = np.array(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.float32)
    kinases = np.array(kinases)
    kinases = torch.tensor(kinases, dtype=torch.float32)

    print(f"Total sequences: {len(peps)}")
    print(f"Label shape: {labels.shape}")
    return PTMDataset_finetune_binary(peps, labels, tokenizer, seq_lens, max_length=max_pep_len, kinases=kinases)


def load_data_finetune_omni(csv_file, tokenizer, max_len=50):
    # Read CSV file
    df = pd.read_csv(csv_file)

    peps = df['peptide'].tolist()
    kinase_names = df['kin_id'].tolist()
    seq_lens = [len(seq) for seq in peps]
    max_pep_len = max(seq_lens)
    print(f"Max peptide length: {max_pep_len}")
    labels = df['label'].tolist()
    # ptm_types = df['modification'].tolist()
    
    kinases = []

    with open('../data/kinases_dict.json', 'r') as f:
        kinase_dict = json.load(f)
        
    kinase2idx = {v: k for k, v in kinase_dict.items()}
    
    for kinase_name in kinase_names:
        kinase_id = int(kinase2idx[kinase_name.upper()])
        kinases.append(kinase_id)

    labels = np.array(labels)
    labels = torch.tensor(labels, dtype=torch.float32)
    seq_lens = np.array(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.float32)
    kinases = np.array(kinases)
    kinases = torch.tensor(kinases, dtype=torch.float32)

    print(f"Total sequences: {len(peps)}")
    print(f"Label shape: {labels.shape}")
    return PTMDataset_finetune_binary(peps, labels, tokenizer, seq_lens, max_length=max_pep_len, kinases=kinases)


def preprocess_sequence(sequence, window_size=50):
    pep_seq_list = []
    sequence = sequence.upper()
    i = 0
    while i < len(sequence):
        if i + window_size <= len(sequence):
            pep_seq = sequence[i:i + window_size]
        else:
            pep_seq = sequence[i:]
            padding = window_size - len(pep_seq)
            pep_seq = sequence[i:] + 'X' * padding
        
        start = max(0, i - 10)
        end = min(len(sequence), i + window_size + 10)
        pep_seq_extend = sequence[start:end]
        if start == 0:
            pep_seq_extend = 'X' * (10 - i) + pep_seq_extend
        if end == len(sequence):
            pep_seq_extend = pep_seq_extend + 'X' * (10 - (len(sequence) - i - window_size))

        pep_seq_list.append(pep_seq_extend)
        i += window_size 
    return pep_seq_list

def load_data_inference(sequence, tokenizer, max_len=70):
    peps = preprocess_sequence(sequence)
    print(peps)
    # print(f"Processed {len(pep_seq_list)} 50mer peptides from the sequence.")
    seq_lens = [len(seq) for seq in peps]
    max_pep_len = max(seq_lens)
    print(f"Max peptide length: {max_pep_len}")

    ptm_list = [
            'ADP-ribosylation', 'Acetylation', 'Glutathionylation', 'Malonylation',
            'Methylation', 'Phosphorylation', 'S-nitrosylation', 'S-palmitoylation',
            'Succinylation', 'Sulfoxidation', 'Sumoylation', 'Ubiquitination',
            'O-linked Glycosylation', 'Lactylation', 'Hydroxylation', 'Crotonylation',
            'Glutarylation', 'Lactoylation', 'Citrullination', 'Neddylation', 'Formylation',
            'N-linked Glycosylation', 'Amidation', 'Dephosphorylation', 'Myristoylation',
            'Oxidation', 'Pyrrolidone carboxylic acid', 'Sulfation', 'GPI-anchor',
            'Gamma-carboxyglutamic acid', 'C-linked Glycosylation'
        ]

    high_freq_labels = [1, 4, 5, 8, 11, 12, 21]
    high_freq_ptm_list = [ptm_list[i] for i in high_freq_labels]
    ptm2idx = {ptm: i for i, ptm in enumerate(high_freq_ptm_list)}
    seq_lens = [length - 20 for length in seq_lens]
    seq_lens = np.array(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.float32)

    print(f"Total sequences: {len(peps)}")

    return PTMDataset_(peps, None, tokenizer, seq_lens, max_length=max_len)


def load_data_finetune_inference(pep, kin_id, tokenizer):
    peps = [pep]
    kinase_names = [kin_id]
    seq_lens = [len(seq) for seq in peps]
    max_pep_len = max(seq_lens)
    print(f"Max peptide length: {max_pep_len}")
    labels = [0]  # dummy label for inference

    kinases = []

    with open('../data/kinases_dict.json', 'r') as f:
        kinase_dict = json.load(f)
    
    kinase2idx = {v: k for k, v in kinase_dict.items()}
    
    for kinase_name in kinase_names:
        kinase_id = int(kinase2idx[kinase_name.upper()])
        kinases.append(kinase_id)

    labels = np.array(labels)
    labels = torch.tensor(labels, dtype=torch.float32)
    seq_lens = np.array(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.float32)
    kinases = np.array(kinases)
    kinases = torch.tensor(kinases, dtype=torch.float32)

    return PTMDataset_finetune_binary(peps, labels, tokenizer, seq_lens, max_length=max_pep_len, kinases=kinases)
