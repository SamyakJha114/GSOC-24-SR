import random
import os
import numpy as np
import torch
import pandas as pd
from argparse import ArgumentParser
from data_preprocessing import load_data,preprocess_data
from Tokenizers import Encoder_tokeniser, DecoderTokenizer
from SymbolicDPOTrainer import SymbolicDPOTrainer
from model import Model_seq2seq
from Config import Config

def main(config, file_index):
    # Set random seeds for reproducibility
    random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Initialize tokenizers
    encoder_tokenizer = Encoder_tokeniser(2, 1, 100, config.encoder_vocab_path)
    decoder_tokenizer = DecoderTokenizer(config.decoder_vocab_path)

    # Load and preprocess data
    points, num_vars = load_data(config, file_index)
    train_df, df_target, datasets = preprocess_data(config)

    # Initialize and load the model
    model = Model_seq2seq(num_encoder_layers=2,
                num_decoder_layers=6,
                emb_size=64,
                nhead=8,
                src_vocab_size=1104,
                tgt_vocab_size=59,
                input_emb_size=64,
                max_input_points=33,
                device=config.device,
                )

    reference_model = Model_seq2seq(num_encoder_layers=2,
                num_decoder_layers=6,
                emb_size=64,
                nhead=8,
                src_vocab_size=1104,
                tgt_vocab_size=59,
                input_emb_size=64,
                max_input_points=33,
                device=config.device,
                )
    path = config.model_weights_path
    model.load_state_dict(torch.load(path)["state_dict"])
    reference_model.load_state_dict(torch.load(path)["state_dict"])

    trainer = SymbolicDPOTrainer(
        model = model, 
        reference_model = reference_model, 
        decoder_tokenizer = decoder_tokenizer, 
        dataset = datasets['test'],
        num_vars = num_vars, 
        points = points,
        file_index = file_index)

    trainer.training_loop()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file_index', type=int, help='which test file to run on')
    args = parser.parse_args()

    config = Config()

    main(config, args.file_index)
