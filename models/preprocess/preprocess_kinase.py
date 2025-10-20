import pandas as pd
import torch
import esm
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModel

# 配置参数
CSV_PATH = "../data/omnipath_kinase_id_name_seqs.csv"
SAVE_PATH = "../data/kinase_embeddings_150m.npz"
ESM_MODEL_NAME = "esm2_t30_150M_UR50D"  # 较小的高效模型
BATCH_SIZE = 4  # 根据GPU内存调整
DEVICE = "cuda:6" if torch.cuda.is_available() else "cpu"

# 1. 数据预处理（添加 Alphabet 参数）
def load_and_preprocess(csv_path, alphabet):
    df = pd.read_csv(csv_path)
    standard_aa = alphabet.standard_toks

    valid_seqs = []
    for seq in df["seq"]:
        seq_upper = seq.upper()
        if all(aa in standard_aa for aa in seq_upper):
            valid_seqs.append(seq_upper)
        else:
            print(f"Find invalid sequence and removed: {seq}")
    
    df["seq"] = valid_seqs
    return df

def load_esm_model(model_name):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    return model.to(DEVICE), batch_converter, alphabet

def generate_embeddings(model, batch_converter, sequences):
    model.eval()
    embeddings = {}
    for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc="Processing"):
        batch_seqs = sequences[i:i+BATCH_SIZE]
        batch_labels, batch_strs, batch_tokens = batch_converter(
            [("seq", seq) for seq in batch_seqs]
        )
        batch_tokens = batch_tokens.to(DEVICE)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
        token_embeddings = results["representations"][model.num_layers]
        sequence_embeddings = token_embeddings[:, 1:-1, :].mean(dim=1)
        for seq, emb in zip(batch_seqs, sequence_embeddings.cpu().numpy()):
            embeddings[seq] = emb
    return embeddings

if __name__ == "__main__":
    model, converter, alphabet = load_esm_model(ESM_MODEL_NAME)
    print(f"Loaded ESM model: {ESM_MODEL_NAME}")
    
    df = load_and_preprocess(CSV_PATH, alphabet)
    kinase_dict = {}
    for i in range(len(df)):
        kinase_dict[str(i)] = df['kinase'][i]
    kinase_dict[len(df)] = 'UNMENTIONED'

    import json
    with open('../data/kinases_dict.json', 'w') as f:
        json.dump(kinase_dict, f)

    sequence_list = df["seq"].tolist()
    embeddings = generate_embeddings(model, converter, sequence_list)
    
    kinase_to_emb = {row["kinase"]: embeddings[row["seq"]] for _, row in df.iterrows()}
    np.savez_compressed(SAVE_PATH, **kinase_to_emb)
    print(f"保存完成！嵌入维度：{next(iter(kinase_to_emb.values())).shape}")

    # test load
    # embeddings = np.load("../data/kinase_embeddings.npz")
    # kinase_name = 'LYN'
    # pkc_emb = embeddings[kinase_name]
    # print(pkc_emb)