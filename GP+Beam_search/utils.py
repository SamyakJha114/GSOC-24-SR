import torch
import re
from Tokenizers import convert_to_functional_form

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