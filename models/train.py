import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from config_pep import config
from dataset import load_data_, load_data_finetune_binary, load_data_finetune_omni
from model import AMPPredictor, create_model_finetune, create_model_binary, create_model_trans_bias
from loss import HybridMacroMicroLoss, FinetuneBinaryLoss
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
import wandb
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import esm
import argparse
import os
import random
import numpy as np
from transformers import AutoTokenizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import label_binarize

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Setting random seed as {seed}.")

high_frequency_indices = [1, 4, 5, 8, 11, 12, 21]
select_num_class = len(high_frequency_indices)+1
config.num_labels = select_num_class
print("select_num_class: ", select_num_class)
max_seq_len = config.max_seq_len


def train(model, train_loader, optimizer, criterion, device, version="v2", centroids=None, mlp=None):
    model.train()
    total_loss = 0.0
    
    if version == "stage1":
        class_weights_ = None
        for token, labels, mask, seq_lens, sequence in tqdm(train_loader):
            token, labels, mask, seq_lens = token.to(device), labels.to(device), mask.to(device), seq_lens.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=token, attention_mask=mask, protein_sequences=sequence, return_dict=True)
            if len(outputs.shape) == 4:
                dim_1, dim_2, dim_3, dim_4 = outputs.shape
                outputs = outputs.view(dim_1 * dim_3, dim_4)
            elif len(outputs.shape) == 3:
                dim_1, dim_2, dim_3 = outputs.shape
                outputs = outputs.view(dim_1 * dim_2, dim_3)

            batch_size, seq_len, num_classes = labels.shape
            labels = labels.view(batch_size * seq_len, -1)
            loss = criterion(outputs, labels, seq_lens, class_weights=class_weights_)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    elif version == "stage2":
        for token, labels, mask, seq_lens, kinase, sequences in tqdm(train_loader):
            token, labels, mask, seq_lens, kinase = token.to(device), labels.to(device), mask.to(device), seq_lens.to(device), kinase.to(device)
            optimizer.zero_grad()
            outputs = model(token, mask, kinase, sequences, return_dict=True)
            loss = criterion(outputs, labels, seq_lens)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device, num_classes=select_num_class, threshold=0.5, version="v1", centroids=None, mlp=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    if version == "stage1":
        with torch.no_grad():
            for token, labels, mask, seq_lens, sequence in tqdm(val_loader):
                token, labels, mask, seq_lens = token.to(device), labels.to(device), mask.to(device), seq_lens.to(device)
                
                outputs = model(token, mask, sequence)
                if len(outputs.shape) == 4:
                    dim_1, dim_2, dim_3, dim_4 = outputs.shape
                    outputs = outputs.view(dim_1 * dim_3, dim_4)
                elif len(outputs.shape) == 3:
                    dim_1, dim_2, dim_3 = outputs.shape
                    outputs = outputs.view(dim_1 * dim_2, dim_3)
                
                batch_size, seq_len, num_classes = labels.shape
                labels = labels.view(batch_size * seq_len, -1)
                
                loss = criterion(outputs, labels, seq_lens)
                
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                mask = torch.zeros((batch_size, max_seq_len-20), device=labels.device)
                for i, length in enumerate(seq_lens):
                    mask[i, :int(length)] = 1
                mask = mask.view(-1, 1).expand(-1, num_classes)
                preds = preds * mask
                labels = labels * mask
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            aucs = []  # AUROC
            auprcs = []
            mccs = []
            
            for i in range(num_classes):
                acc = accuracy_score(all_labels[:, i], all_preds[:, i])
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    prec, rec, f1, _ = precision_recall_fscore_support(all_labels[:, i], all_preds[:, i], pos_label=1, average='binary', zero_division=0)
                    # prec, rec, f1, _ = precision_recall_fscore_support(all_labels[:, i], all_preds[:, i], pos_label=1, average='micro', zero_division=0)
                
                try:
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                except ValueError:
                    auc = np.nan
                
                # AUPRC
                try:
                    auprc = average_precision_score(all_labels[:, i], all_preds[:, i])
                except ValueError:
                    auprc = np.nan
                    
                # mcc
                try:
                    mcc = matthews_corrcoef(all_labels[:, i], all_preds[:, i])
                except ValueError:
                    mcc = np.nan
                    
                accuracies.append(acc)
                precisions.append(prec)
                recalls.append(rec)
                f1_scores.append(f1)
                aucs.append(auc)
                auprcs.append(auprc)
                mccs.append(mcc)

                if config.use_wandb:
                    log_dict = {
                        f'val_accuracy_class_{i}': acc,
                    }
                    if not np.isnan(prec):
                        log_dict[f'val_precision_class_{i}'] = prec
                    if not np.isnan(rec):
                        log_dict[f'val_recall_class_{i}'] = rec
                    if not np.isnan(f1):
                        log_dict[f'val_f1_class_{i}'] = f1
                    if not np.isnan(auc):
                        log_dict[f'val_auc_class_{i}'] = auc
                    if not np.isnan(auprc):
                        log_dict[f'val_auprc_class_{i}'] = auprc
                    if not np.isnan(mcc):
                        log_dict[f'val_mcc_class_{i}'] = mcc
                    
                    wandb.log(log_dict)
            
            avg_accuracy = np.nanmean(accuracies)
            avg_precision = np.nanmean(precisions)
            avg_recall = np.nanmean(recalls)
            avg_f1 = np.nanmean(f1_scores)
            avg_auc = np.nanmean(aucs)
            avg_auprc = np.nanmean(auprcs)
            avg_mcc = np.nanmean(mccs)
            
            if config.use_wandb:
                wandb.log({
                    'val_loss': total_loss / len(val_loader),
                    'val_avg_accuracy': avg_accuracy,
                    'val_avg_precision': avg_precision,
                    'val_avg_recall': avg_recall,
                    'val_avg_f1': avg_f1,
                    'val_avg_auc': avg_auc,
                    'val_avg_auprc': avg_auprc,
                    'val_avg_mcc': avg_mcc
                })
            
            return total_loss / len(val_loader), avg_f1

    
    elif version == "stage2":
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for token, labels, mask, seq_lens, kinase, sequences in tqdm(val_loader):
                token, labels, mask, seq_lens, kinase = token.to(device), labels.to(device), mask.to(device), seq_lens.to(device), kinase.to(device)
                
                outputs = model(token, mask, kinase, sequences)
    
                loss = criterion(outputs, labels, seq_lens)
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)        
        binary_acc = accuracy_score(all_labels, all_preds > 0.5)
        binary_prec, binary_rec, binary_f1, _ = precision_recall_fscore_support(all_labels, all_preds > 0.5, average='binary', zero_division=0)
        binary_auc = roc_auc_score(all_labels, all_preds)
        binary_auprc = average_precision_score(all_labels, all_preds)

        print("binary_acc: ", binary_acc)
        print("binary_prec: ", binary_prec)
        print("binary_rec: ", binary_rec)
        print("binary_f1: ", binary_f1)
        print("binary_auc: ", binary_auc)
        print("binary_auprc: ", binary_auprc)
        
        if config.use_wandb:
            wandb.log({
                'val_loss': total_loss / len(val_loader),
                'val_binary_accuracy': binary_acc,
                'val_binary_precision': binary_prec,
                'val_binary_recall': binary_rec,
                'val_binary_f1': binary_f1,
                'val_binary_auc': binary_auc,
                'val_binary_auprc': binary_auprc
            })
        return total_loss / len(val_loader), binary_acc
    

def train_model(version="stage1"):
    print('-'*40)
    print(f"Training model version: {version}")
    print('-'*40)

    train_centroid = None
    val_centroid = None
    if version == "stage1":
        tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

        train_data = load_data_(config.train_file, tokenizer, max_seq_len)
        val_data = load_data_(config.valid_file, tokenizer, max_seq_len)
    
    elif version == "stage2":
        tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
        
        train_data = load_data_finetune_omni(config.train_file, tokenizer)
        val_data = load_data_finetune_omni(config.valid_file, tokenizer)
            
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    
    if version == "stage1":
        model = create_model_trans_bias(mode='lora',model_checkpoint=config.model_checkpoint,num_classes=config.num_labels).to(config.device) # full, lora, last_layer
        mlp = None
    elif version == "stage2":
        model = create_model_binary(mode='lora', checkpoint=config.checkpoint, model_checkpoint=config.model_checkpoint).to(config.device)
        mlp = None
   
    if version == "stage1":
        criterion = HybridMacroMicroLoss(num_classes=config.num_labels)
        criterion_val = HybridMacroMicroLoss(num_classes=config.num_labels)
        print("Successfully loading the training centroids to loss function.")
    elif version == "stage2":
        criterion = FinetuneBinaryLoss()
        criterion_val = FinetuneBinaryLoss()  
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()}
    ], lr=config.learning_rate)
    
    if config.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project)
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(config.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, config.device, version=version, centroids=train_centroid, mlp=mlp)
        val_loss, f1 = validate(model, val_loader, criterion_val, config.device, config.num_labels, version=version, centroids=val_centroid, mlp=mlp)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {f1:.4f}")
        
        if config.use_wandb:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        
        if f1 > best_val_f1:
            best_val_loss = val_loss
            best_val_f1 = f1
            best_epoch = epoch
            torch.save(model.state_dict(), config.model_save_path + f'best_model_{version}.pth')
    
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    print("Training model with a specified version.")
    parser = argparse.ArgumentParser(description='Train a model with a specified version.')
    
    parser.add_argument('--version', type=str, default='stage1',
                        help='Version of the model to train (default: stage1)')
    print("Version of the model to train (default: stage1)")
    args = parser.parse_args()

    print(f"Training model with version: {args.version}")
    train_model(args.version)
