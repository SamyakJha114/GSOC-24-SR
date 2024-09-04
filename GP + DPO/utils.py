import torch
from torch.utils.data import Dataset
import numpy as np
import re
from Tokenizers import convert_to_functional_form ,sympy_to_prefix,convert_to_sympy_expression 

def generate_seed_expressions(dataset,indices,file_index,device,model,decoder_tokenizer):
    seed_expr = []
    for i in indices:
        src = dataset[file_index * 1000 + i][0].unsqueeze(0).to(device)
        src_seq_len = src.shape[1]
        model = model.to(device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device= device).type(torch.bool)
        src_padding_mask = (torch.zeros((src.shape[0], src_seq_len), device= device)).type(torch.bool)
        eq = model.beam_search(src, src_mask, src_padding_mask)

        for j in range(0, 3):
            try:
                b = decoder_tokenizer.equation_decoder([decoder_tokenizer.decode((eq[j][0].to('cpu')).numpy())[0][1:-1]])[0]
                b = convert_to_functional_form(b)
                seed_expr.append(b)          
            except:
                    continue

    protected = {
        'exp': 'protected_exp',
        'div': 'protected_div',
        'sqrt': 'protected_sqrt',
        'pow': 'protected_pow'
    }

    for i in range(len(seed_expr)):
        for a in protected:
            seed_expr[i] = re.sub(a, protected[a], seed_expr[i])
    print(f"Length of seed expression array :- {len(seed_expr)}")

    return seed_expr

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def check_bad(individual):
    try:
        check = sympy_to_prefix(convert_to_sympy_expression(str(individual)))
        return False
    except:
        return True

def freeze_reference_model(reference_model):
    for param in reference_model.parameters():
        param.requires_grad = False

class PreferenceDataset(Dataset):
    def __init__(self, preference_pairs,decoder_tokenizer,max_length=40, padding_idx=0):
        self.pairs = preference_pairs
        self.max_length = max_length
        self.padding_idx = padding_idx
        self.decoder_tokenizer = decoder_tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        better, worse = self.pairs[idx]

        better_expr = sympy_to_prefix(convert_to_sympy_expression(str(better)))
        better_expr = self.decoder_tokenizer.encode([['<bos>'] + better_expr + ['<eos>']])[0]
        worse_expr = sympy_to_prefix(convert_to_sympy_expression(str(worse)))
        worse_expr = self.decoder_tokenizer.encode([['<bos>'] + worse_expr + ['<eos>']])[0]

        # Pad sequences to the maximum length
#         better_expr = self.pad_sequence(better_expr)
#         worse_expr = self.pad_sequence(worse_expr)

        return torch.Tensor(better_expr.astype(np.int32)).long(), torch.Tensor(worse_expr.astype(np.int32)).long()

    # def pad_sequence(self, sequence):
    #     if len(sequence) < self.max_length:
    #         padded_sequence = np.pad(sequence, (0, self.max_length - len(sequence)), 
    #                                  'constant', constant_values=self.padding_idx)
    #     else:
    #         padded_sequence = sequence[:self.max_length]
    #     return padded_sequence