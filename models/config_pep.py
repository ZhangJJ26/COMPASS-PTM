import torch

class Config:
    #------------- Data -------------
    train_file = '../data/add_10aa/train_data.csv'
    valid_file = '../data/add_10aa/valid_data.csv'
    test_file = '../data/add_10aa/test_data_new.csv'

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
    use_wandb = True
    wandb_project = "PTM-MF"
    
    # ------------- Checkpoint & Output -------------
    model_save_path = "../output/stage1/multi-label/"
    checkpoint = "/data0/lsn1/jjzhang/PTM-Site/output/best_model_v2_8_loss_dice_simfocalbce.pth"
    
    # ------------- Evaluation -------------
    threshold = 0.5  # For multi-label classification

config = Config()
