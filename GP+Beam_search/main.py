import random
import os
import numpy as np
import torch
import pandas as pd
from argparse import ArgumentParser
from data_preprocessing import load_data,preprocess_data
from Tokenizers import Encoder_tokeniser, DecoderTokenizer
from utils import generate_seed_expressions
from deap import tools, algorithms
from gp import make_pset, setup_toolbox,run_gp
from model import Model_seq2seq
from config import Config

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
    model = Model_seq2seq(
        num_encoder_layers=2,
        num_decoder_layers=6,
        emb_size=64,
        nhead=8,
        src_vocab_size=1104,
        tgt_vocab_size=59,
        input_emb_size=64,
        max_input_points=33,
        device=config.device
    )
    model.load_state_dict(torch.load(config.model_weights_path)["state_dict"])

    # Generate seed expressions
    random_numbers = [random.randint(0, 999) for _ in range(25)]
    seed_expr = generate_seed_expressions(
        (datasets['test'], random_numbers, file_index, config.device, model, decoder_tokenizer)
    )

    if config.random_init :
        seed_expr = []
    # Genetic Programming setup
    pset = make_pset(num_vars)
    toolbox = setup_toolbox(pset, points)
    run_gp(toolbox, points, seed_expr)

if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument('--file_index', type=int, required=True, help='Index of the file to use')
    # args = parser.parse_args()

    config = Config()
    for i in range(0,10):
        print("Currently running file index :- ",i)
        main(config, i)
