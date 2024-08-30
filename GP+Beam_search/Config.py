import torch

class Config:
    def __init__(self):
        self.seed = 42
        self.encoder_vocab_path = '/kaggle/input/gsoc-symba-task/encoder_vocab (1).txt'
        self.decoder_vocab_path = '/kaggle/input/gsoc-symba-task/decoder_vocab (2).txt'
        self.input_path = '/kaggle/input/gsoc-symba-task/Feynman_with_units/Feynman_with_units/'
        self.test_file_paths = [
            'I.6.2a', 'I.12.5', 'I.18.14', 'I.39.1', 'I.43.16',
            'I.43.31', 'II.4.23', 'II.21.32', 'II.35.21', 'II.38.3'
        ]
        self.train_df_path = '/kaggle/input/gsoc-dataset-arrays/train_df.csv'
        self.df_target_path = '/kaggle/input/gsoc-symba-task/FeynmanEquations.csv'
        self.dataset_arrays_path = '/kaggle/input/gsoc-dataset-arrays/dataset_arrays/'
        self.model_weights_path = '/kaggle/input/symba-dpo-trained-model/model_weights.pth'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'