import torch.nn as nn
from config_pep import config
import torch
import esm
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForTokenClassification, AutoModel, AutoConfig, EsmConfig
from transformers.models.esm.modeling_esm import EsmSelfAttention
import json
import numpy as np
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
import math
import torch
import torch.nn as nn
import os
from types import SimpleNamespace
import traceback
import inspect

class ResidualBlock_v2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()

        self.downsample = None
        if in_dim != out_dim:
            self.downsample = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.norm(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AMPPredictor(nn.Module):
    def __init__(self, num_labels=8):
        super().__init__()
        
        layers = []
        in_dim = config.mlp_hidden_dims[0]
        
        for hidden_dim in config.mlp_hidden_dims[1:-1]:
            layers.append(ResidualBlock_v2(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_labels))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
    
    def inference(self, x):
        intermediate_layers = list(self.mlp.children())[:-1]
        intermediate_model = nn.Sequential(*intermediate_layers)
        
        with torch.no_grad():
            return intermediate_model(x)
        

AA_PHYCHEM = {
    # amino acid: [molecular weight, pI, hydrophobicity, volume]
    'A': [89.09, 6.0, 1.8, 0.7],    # 丙氨酸 (Alanine)
    'R': [174.20, 10.8, -4.5, 4.2], # 精氨酸 (Arginine)
    'N': [132.12, 5.4, -3.5, 3.2],  # 天冬酰胺 (Asparagine)
    'D': [133.10, 2.8, -3.5, 4.4],  # 天冬氨酸 (Aspartic acid)
    'C': [121.16, 5.1, 2.5, 1.5],   # 半胱氨酸 (Cysteine)
    'Q': [146.15, 5.7, -3.5, 3.7],  # 谷氨酰胺 (Glutamine)
    'E': [147.13, 3.2, -3.5, 4.4],  # 谷氨酸 (Glutamic acid)
    'G': [75.07, 5.9, -0.4, 0.7],   # 甘氨酸 (Glycine)
    'H': [155.16, 7.6, -3.2, 3.0],  # 组氨酸 (Histidine)
    'I': [131.18, 6.0, 4.5, 0.7],   # 异亮氨酸 (Isoleucine)
    'L': [131.18, 6.0, 3.8, 0.7],   # 亮氨酸 (Leucine)
    'K': [146.19, 9.7, -3.9, 4.0],  # 赖氨酸 (Lysine)
    'M': [149.21, 5.7, 1.9, 0.7],   # 甲硫氨酸 (Methionine)
    'F': [165.19, 5.5, 2.8, 0.3],   # 苯丙氨酸 (Phenylalanine)
    'P': [115.13, 6.3, -1.6, 1.9],  # 脯氨酸 (Proline)
    'S': [105.09, 5.7, -0.8, 2.1],  # 丝氨酸 (Serine)
    'T': [119.12, 5.6, -0.7, 2.1],  # 苏氨酸 (Threonine)
    'W': [204.23, 5.9, -0.9, 1.4],  # 色氨酸 (Tryptophan)
    'Y': [181.19, 5.7, -1.3, 1.6],  # 酪氨酸 (Tyrosine)
    'V': [117.15, 6.0, 4.2, 0.7],   # 缬氨酸 (Valine)
    # special tokens
    '<cls>': [0.0, 0.0, 0.0, 0.0],
    '<pad>': [0.0, 0.0, 0.0, 0.0],
    '<unk>': [0.0, 0.0, 0.0, 0.0]
}
class PhysChemEmbedder:
    def __init__(self):
        self._build_mselfapping()
    
    def _build_mselfapping(self):
        all_features = [v for k,v in AA_PHYCHEM.items() if k not in ['<cls>','<pad>','<unk>']]
        self.scaler = StandardScaler().fit(all_features)

        self.embedding_dict = {}
        for aa, feat in AA_PHYCHEM.items():
            scaled = self.scaler.transform([feat])[0] if aa not in ['<cls>','<pad>','<unk>'] else feat
            self.embedding_dict[aa] = scaled
            
    def __call__(self, aa_sequence):
        """输入氨基酸序列，输出化学特征矩阵"""
        return np.array([self.embedding_dict.get(aa, self.embedding_dict['<unk>']) 
                       for aa in aa_sequence])


class fusion_model_binary_2(nn.Module):
    def __init__(self, pretrained_model, mlp_binary):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.mlp_binary = mlp_binary
        # load kinase embeddings
        with open('../data/kinases_dict.json', 'r') as f:
            self.kinases_dict = json.load(f)
        self.kinase_embeddings = np.load("../data/kinase_embeddings_150m.npz")
        embed_dim = 640

        self.gate_substrate = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*2, embed_dim),
            torch.nn.Sigmoid()
        )
        self.gate_kinase = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*2, embed_dim),
            torch.nn.Sigmoid()
        )
        self.res_info_composer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*2, embed_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim*2, embed_dim),
        )

        self.a = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.empha = nn.Embedding(1, embed_dim)
        
        
    def forward(self, input_ids, mask, kinase_ids, sequences, return_dict=None, return_embedding=False):
        outputs = self.pretrained_model(
            input_ids,
            mask,
            sequences
        )
        k_mer = 7
        last_hidden = outputs

        kinase_embs = torch.zeros_like(last_hidden)
        for i in range(kinase_ids.size(0)):
            kinase_id = kinase_ids[i]
            kinase_name = self.kinases_dict.get(str(int(kinase_id.item())), 'None')
            if kinase_name == 'None' or kinase_name == 'unmentioned' or kinase_name == 'UNMENTIONED':   
                continue
            kinase_emb = self.kinase_embeddings[kinase_name]
            kinase_emb = torch.tensor(kinase_emb).to(kinase_ids.device)
            # expand to match the shape of last_hidden
            kinase_emb = kinase_emb.unsqueeze(0).expand(last_hidden.size(1), -1)
            kinase_embs[i] = kinase_emb

        last_hidden[:, k_mer, :] += self.empha.weight[0]
        
        fusion_emb = torch.cat([last_hidden, kinase_embs], dim=-1)
        batch_size, seq_len, _ = fusion_emb.shape
        fusion_flat = fusion_emb.view(batch_size * seq_len, -1)
        
        # generate gates and residuals
        gate_substrate = self.gate_substrate(fusion_flat)  # [batch*seq, embed_dim]
        gate_kinase = self.gate_kinase(fusion_flat)        # [batch*seq, embed_dim]
        residual = self.res_info_composer(fusion_flat)    # [batch*seq, embed_dim]
        gate_substrate = gate_substrate.view(batch_size, seq_len, -1)
        gate_kinase = gate_kinase.view(batch_size, seq_len, -1)
        residual = residual.view(batch_size, seq_len, -1)
        
        fusion_emb = (gate_substrate * last_hidden * self.a[0] + 
                      gate_kinase * kinase_embs * self.a[1] + 
                      residual * self.a[2])
        fusion_emb = fusion_emb[:, k_mer].squeeze(1)

        if return_embedding:
            return fusion_emb
        else:
            return self.mlp_binary(fusion_emb)
    

def create_model_binary(mode='full', checkpoint="/home/student/Documents/jingjie/research/PTM-site/output/best_model_v2_8_high_aa.pth", model_checkpoint="facebook/esm2_t6_8M_UR50D", num_classes=2):
    # model = create_model(mode='lora').to(config.device) # full, lora, last_layer
    model = create_model_trans_bias(mode='lora',model_checkpoint=model_checkpoint).to(config.device)
    mlp_binary = AMPPredictor(num_labels=1).to(config.device)
    mlp_multi = AMPPredictor(num_labels=num_classes).to(config.device)
    
    # zjj_finetune_ablation_study
    state_dict = torch.load(checkpoint, map_location=config.device)
    model.load_state_dict(state_dict)
    # return model
    # zjj_finetune_omni_new
    # # 冻住模型参数
    # for param in model.esm.parameters():
    #     param.requires_grad = False
    # 设计finetune的模型
    finetune_model = fusion_model_binary_2(model, mlp_binary)
    # finetune_model = fusion_model_2(model, mlp_binary, mlp_multi)
    return finetune_model


class LoRAESMWithTransformer(nn.Module):
    """
    LoRA ESM model with custom Transformer layers for PTM site prediction
    """
    def __init__(self, esm_model, num_custom_layers=2, hidden_size=360, crosstalk_matrix=None, esm_mlp=None,model_checkpoint="facebook/esm2_t6_8M_UR50D"):
        super().__init__()
        self.esm = esm_model
        esm_config = EsmConfig.from_pretrained(model_checkpoint)
        self.aa2clm = torch.load("aa2selfies_embeddings.pt")
        self.custom_layers = nn.ModuleList([
            CustomTransformerLayer(esm_config, crosstalk_matrix, mlp=esm_mlp)
            for _ in range(num_custom_layers)
        ])
        hidden_size_all = hidden_size+320+4  # ESM hidden size + CLM size + 4 chemical features

        self.concat_linear = nn.Linear(hidden_size_all, hidden_size)
        self.chem_embedder = PhysChemEmbedder()
        self.gated_feature_composer = nn.Sequential(
            nn.BatchNorm1d(hidden_size_all),
            nn.ReLU(),
            nn.Linear(hidden_size_all, hidden_size)
        )
        self.res_info_composer = nn.Sequential(
            nn.BatchNorm1d(hidden_size_all),
            nn.ReLU(),
            nn.Linear(hidden_size_all, hidden_size_all),
            nn.ReLU(),
            nn.Linear(hidden_size_all, hidden_size)
        )
        self.a = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        
        self.classifier = AMPPredictor()

    def forward(self, input_ids, attention_mask, protein_sequences, return_dict=None):
        for param in self.esm.parameters():
            param.requires_grad = False
        esm_outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        esm_hidden = esm_outputs.hidden_states[-1]
        esm_hidden = esm_hidden[:, 1:-1]
        protein_sequences = protein_sequences
        
        batch_chem = []
        batch_clm_features = []
        for seq in protein_sequences:
            seq_clm_features = []
            for i, aa in enumerate(seq):
                if aa in self.aa2clm:
                    clm = self.aa2clm[aa]
                    seq_clm_features.append(clm.to(esm_hidden.device))
                else:
                    seq_clm_features.append(torch.zeros(320).to(esm_hidden.device))
            seq_clm_features = torch.stack(seq_clm_features)
            batch_clm_features.append(seq_clm_features)
            chem_feat = self.chem_embedder(seq)
            chem_feat = torch.tensor(chem_feat, dtype=torch.float32).to(input_ids.device)
            batch_chem.append(chem_feat)
        clm_features = torch.stack(batch_clm_features)
        chem_features = torch.stack(batch_chem)

        seq_emb = torch.cat([esm_hidden, clm_features], dim=-1)
        add_phychem = torch.cat([seq_emb, chem_features], dim=-1)
        batch_size, seq_len, _ = add_phychem.shape
        add_phychem = add_phychem.view(batch_size * seq_len, -1)
        f1 = self.gated_feature_composer(add_phychem)
        f1 = f1.view(batch_size, seq_len, -1)
        f2 = self.res_info_composer(add_phychem)
        f2 = f2.view(batch_size, seq_len, -1)
        fused = F.sigmoid(f1) * esm_hidden * self.a[0] + f2 * self.a[1]
        
        attention_mask = attention_mask[:, 1:-1]
        attention_mask = attention_mask[:, 10:-10]
        prompt_fused = fused[:, 10:-10]
        for layer in self.custom_layers:
            prompt_fused = layer(prompt_fused, attention_mask)
        return self.classifier(prompt_fused)
 

def create_model_trans_bias(mode='full', model_checkpoint="facebook/esm2_t6_8M_UR50D",num_classes=8):
    base_model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["attention.self.query", "attention.self.key", "attention.self.value"],
        bias="none",
    )
    
    if mode == "lora":
        lora_esm = get_peft_model(base_model, lora_config)

    else:
        lora_esm = base_model

    esm_mlp = AMPPredictor(num_labels=num_classes)
    crosstalk_matrix = torch.load("matrix.pt")

    full_model = LoRAESMWithTransformer(
        esm_model=lora_esm,
        num_custom_layers=2,
        hidden_size=base_model.config.hidden_size,
        crosstalk_matrix=crosstalk_matrix,
        esm_mlp=esm_mlp,
        model_checkpoint=model_checkpoint
    )
    
    for param in full_model.custom_layers.parameters():
        param.requires_grad_(True)
    for param in full_model.gated_feature_composer.parameters():
        param.requires_grad_(True)
    for param in full_model.res_info_composer.parameters():
        param.requires_grad_(True)
    for param in full_model.classifier.parameters():
        param.requires_grad_(True)

    return full_model


class CustomTransformerLayer(nn.Module):
    def __init__(self, config, crosstalk_matrix, mlp=None):
        super().__init__()
        self.self_attention = EsmSelfAttention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        assert crosstalk_matrix.shape[0] == crosstalk_matrix.shape[1], "crosstalk_matrix must be square"
        self.linear_1 = nn.Linear(crosstalk_matrix.shape[0], 8)
        self.linear_2 = nn.Linear(crosstalk_matrix.shape[0], 8)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.zeros_(self.linear_2.bias)
        self.crosstalk_matrix = nn.Parameter(crosstalk_matrix.float())
        self.bias_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        self.ptm_predictor = mlp
        
        self.linear_attention = nn.Linear(6400, 320)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(self, hidden_states, attention_mask=None):
        output_1 = self.linear_1(self.crosstalk_matrix.T).T  # [8, R] -> [R,8]
        output_2 = self.linear_2(self.crosstalk_matrix)     # [R,8]
        crosstalk_R = torch.matmul(output_1, output_2)      # [8,8]
        
        ptm_probs = F.softmax(self.ptm_predictor(hidden_states), dim=-1)

        B = torch.einsum('blc,cd,bmd->blm', ptm_probs, crosstalk_R, ptm_probs)
        B = self.dropout(torch.tanh(B)) * self.bias_scale

        query = self.self_attention.query(hidden_states)
        key = self.self_attention.key(hidden_states)
        value = self.self_attention.value(hidden_states)

        num_heads = self.self_attention.num_attention_heads
        head_dim = self.self_attention.attention_head_size
        batch_size, seq_len, _ = hidden_states.shape

        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        attention_scores = attention_scores + B.unsqueeze(1)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)
        context = self.dropout(context)
        hidden_states = self.layer_norm(hidden_states + context)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        return hidden_states