import torch

class Config:
    version = "stage2"
    #------------- Data -------------
    # ============ stage1: multi-label classification ============
    if version == "stage1":
        train_file = '../data/stage1/train_data.csv'
        valid_file = '../data/stage1/valid_data.csv'
        test_file = '../data/stage1/test_data_new.csv'
    # ============ stage2: enzyme-substrate pairing ============
    elif version == "stage2":
        train_file = '../data/stage2/kinase/train_kin.csv'
        valid_file = '../data/stage2/kinase/valid_kin.csv'
        test_file = '../data/stage2/kinase/test_kin.csv'

    # ------------- Model -------------
    esm2_model = "esm2_t30_150M_UR50D"
    mlp_hidden_dims = [640, 512, 64]
    model_checkpoint = "facebook/esm2_t30_150M_UR50D"

    num_labels = 31
    v3_feat_dim = mlp_hidden_dims[-1]
    
    # ------------- Training -------------
    batch_size = 1024
    learning_rate = 2e-5
    num_epochs = 200
    save_interval = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size_pre = 4
    max_seq_len = 70
    early_stop = True
    patience = 20 
    use_wandb = False
    wandb_project = "PTM-MF"
    seed = 42
    
    # ------------- Checkpoint & Output -------------
    model_save_path = "../output/stage1/multi-label/"
    checkpoint = "../checkpoint/best_stage1.pth"
    
    # ------------- Evaluation -------------
    threshold = 0.5  # For multi-label classification

config = Config()
