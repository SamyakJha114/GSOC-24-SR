import torch

class Config:
    def __init__(self,args):
        self.seed = 42
        self.encoder_vocab_path = '/pscratch/sd/s/samyak09/encoder_vocab (1).txt'
        self.decoder_vocab_path = '/pscratch/sd/s/samyak09/decoder_vocab (2).txt'
        self.input_path = '/pscratch/sd/s/samyak09/Feynman_with_units/Feynman_with_units'
        self.test_file_paths = [
            'I.6.2a', 'I.12.5', 'I.18.14', 'I.39.1', 'I.43.16',
            'I.43.31', 'II.4.23', 'II.21.32', 'II.35.21', 'II.38.3'
        ]
        self.finetune_file_paths = [
        'III.21.20','I.32.5','I.18.4','II.38.14','I.44.4',
        'I.12.11','I.47.23','II.10.9','II.11.27',
        ]
        self.train_df_path = '/pscratch/sd/s/samyak09/train_df.csv'
        self.df_target_path = '/pscratch/sd/s/samyak09/FeynmanEquationsModified.csv'
        self.dataset_arrays_path = '/pscratch/sd/s/samyak09/dataset_arrays'
        self.model_weights_path = '/pscratch/sd/s/samyak09/default/best_checkpoint.pth'
        self.random_init = args.random_init
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'