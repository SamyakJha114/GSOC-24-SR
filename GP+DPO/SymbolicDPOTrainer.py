import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from gp import make_pset, setup_toolbox,run_gp,generate_preference_pairs
from utils import freeze_reference_model,generate_seed_expressions,PreferenceDataset,generate_square_subsequent_mask

PAD_IDX = 0

class SymbolicDPOTrainer:
    def __init__(self, model, reference_model, decoder_tokenizer, dataset, num_vars, points,file_index ,beta=0.001, device='cuda'):
        self.model = model
        self.reference_model = reference_model
        freeze_reference_model(self.reference_model)
        self.decoder_tokenizer = decoder_tokenizer
        self.dataset = dataset
        self.num_vars = num_vars
        self.points = points
        self.beta = beta
        self.device = device
        self.src = None
        self.file_index = file_index
        self.toolbox = None
        self.reference_model = self.reference_model.to(self.device)
        self.model = self.model.to(self.device)
    
    def dpo_loss(self, pi_logps, ref_logps):
        pi_logratios = pi_logps[:, 0] - pi_logps[:, 1]
        ref_logratios = ref_logps[:, 0] - ref_logps[:, 1]
        losses = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios))
        rewards = self.beta * (pi_logps - ref_logps).detach()
        return losses, rewards
    def train_transformer_dpo(self, preference_pairs, epochs=10, batch_size=1, lr=1e-4):
        train, test = train_test_split(preference_pairs,test_size = 0.1,random_state = 42)
        train_dataset = PreferenceDataset(train)
        test_dataset = PreferenceDataset(test)
        train_dataloader = DataLoader(train_dataset, batch_size=1,shuffle=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset,batch_size =1,shuffle=False, drop_last=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 4 , gamma=0.05, verbose=True)
        self.model = self.model.to(self.device)
        self.reference_model = self.reference_model.to(self.device)
        self.reference_model.eval()
        for epoch in tqdm(range(epochs)):
            print("TRAINING THE TRANSFORMER")
            self.model.train()
            total_loss = 0
            total_test_loss = 0
            for better, worse in train_dataloader:
                better = better.to(self.device)
                worse = worse.to(self.device)
                optimizer.zero_grad()
                random_num = [random.randint(0, 999) for _ in range(batch_size)]
                for j,i in enumerate(random_num):
                    if j == 0:
                        src = self.dataset[self.file_index * 1000 + i][0].unsqueeze(0).to(self.device)
                    else :
                        temp_src = self.dataset[self.file_index * 1000 + i][0].unsqueeze(0).to(self.device)
                        src = torch.cat([src,temp_src],dim = 0)
                src_mask = torch.zeros((src.shape[1], src.shape[1]), device=self.device).type(torch.bool)
                src_padding_mask = (torch.zeros((src.shape[0], src.shape[1]), device=self.device)).type(torch.bool)
                
                better = better.repeat(batch_size, 1)
                worse = worse.repeat(batch_size, 1)
                
                better_mask = generate_square_subsequent_mask(better[:,1:].shape[1], self.device)
                worse_mask = generate_square_subsequent_mask(worse[:,1:].shape[1], self.device)
                better_padding_mask = (better[:,1:] == PAD_IDX).to(self.device)
                worse_padding_mask = (worse[:,1:] == PAD_IDX).to(self.device)
                
                pi_logps_better = self.model(src, better[:, :-1], 
                                             src_mask,better_mask, 
                                             src_padding_mask,better_padding_mask,
                                             src_padding_mask).log_softmax(dim=-1)
                
                pi_logps_worse = self.model(src, worse[:, :-1], 
                                            src_mask, worse_mask, 
                                            src_padding_mask, worse_padding_mask, 
                                            src_padding_mask).log_softmax(dim=-1)
                
                ref_logps_better = self.reference_model(src, better[:, :-1], 
                                                        src_mask, better_mask, 
                                                        src_padding_mask, better_padding_mask, 
                                                        src_padding_mask).log_softmax(dim=-1)
                
                ref_logps_worse = self.reference_model(src, worse[:, :-1], 
                                                       src_mask, worse_mask, 
                                                       src_padding_mask, worse_padding_mask, 
                                                       src_padding_mask).log_softmax(dim=-1)
                
                pi_logps_better = pi_logps_better.gather(dim=-1, index=better[:,1:].unsqueeze(-1)).squeeze(-1)
                pi_logps_worse = pi_logps_worse.gather(dim=-1, index=worse[:,1:].unsqueeze(-1)).squeeze(-1)
                ref_logps_better = ref_logps_better.gather(dim=-1, index=better[:,1:].unsqueeze(-1)).squeeze(-1)
                ref_logps_worse = ref_logps_worse.gather(dim=-1, index=worse[:,1:].unsqueeze(-1)).squeeze(-1)
                
                
                pi_logps_better = pi_logps_better.sum(dim=1)
                pi_logps_worse = pi_logps_worse.sum(dim=1)
                ref_logps_better = ref_logps_better.sum(dim=1)
                ref_logps_worse = ref_logps_worse.sum(dim=1)
                
                pi_logps = torch.stack([pi_logps_better, pi_logps_worse], dim=1)
                ref_logps = torch.stack([ref_logps_better, ref_logps_worse], dim=1)
                losses, _ = self.dpo_loss(pi_logps, ref_logps)
                
                loss = losses.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.model.eval()
            print("TESTING")
            for better,worse in test_dataloader:
                with torch.no_grad():
                    better = better.to(self.device)
                    worse = worse.to(self.device)
#                     random_num = random.randint(0, 999)
#                     src = self.dataset[self.file_index * 1000 + random_num][0].unsqueeze(0).to(self.device)
                    random_num = [random.randint(0, 999) for _ in range(batch_size)]
                    for j,i in enumerate(random_num):
                        if j == 0:
                            src = self.dataset[self.file_index * 1000 + i][0].unsqueeze(0).to(self.device)
                        else :
                            temp_src = self.dataset[self.file_index * 1000 + i][0].unsqueeze(0).to(self.device)
                            src = torch.cat([src,temp_src],dim = 0)
                            
                    better = better.repeat(batch_size, 1)
                    worse = worse.repeat(batch_size, 1)
                    
                    src_mask = torch.zeros((src.shape[1], src.shape[1]), device=self.device).type(torch.bool)
                    src_padding_mask = (torch.zeros((src.shape[0], src.shape[1]), device=self.device)).type(torch.bool)

                    better_mask = generate_square_subsequent_mask(better[:,1:].shape[1], self.device)
                    worse_mask = generate_square_subsequent_mask(worse[:,1:].shape[1], self.device)
                    better_padding_mask = (better[:,1:] == PAD_IDX).to(self.device)
                    worse_padding_mask = (worse[:,1:] == PAD_IDX).to(self.device)

                    pi_logps_better = self.model(src, better[:,:-1], 
                                                 src_mask,better_mask, 
                                                 src_padding_mask,better_padding_mask,
                                                 src_padding_mask).log_softmax(dim=-1)

                    pi_logps_worse = self.model(src, worse[:,:-1], 
                                                src_mask, worse_mask, 
                                                src_padding_mask, worse_padding_mask, 
                                                src_padding_mask).log_softmax(dim=-1)

                    ref_logps_better = self.reference_model(src, better[:,:-1], 
                                                            src_mask, better_mask, 
                                                            src_padding_mask, better_padding_mask, 
                                                            src_padding_mask).log_softmax(dim=-1)

                    ref_logps_worse = self.reference_model(src, worse[:,:-1], 
                                                           src_mask, worse_mask, 
                                                           src_padding_mask, worse_padding_mask, 
                                                           src_padding_mask).log_softmax(dim=-1)
                    
                    pi_logps_better = pi_logps_better.gather(dim=-1, index=better[:,1:].unsqueeze(-1)).squeeze(-1)
                    pi_logps_worse = pi_logps_worse.gather(dim=-1, index=worse[:,1:].unsqueeze(-1)).squeeze(-1)
                    ref_logps_better = ref_logps_better.gather(dim=-1, index=better[:,1:].unsqueeze(-1)).squeeze(-1)
                    ref_logps_worse = ref_logps_worse.gather(dim=-1, index=worse[:,1:].unsqueeze(-1)).squeeze(-1)
                    
                    pi_logps_better = pi_logps_better.sum(dim=1)
                    pi_logps_worse = pi_logps_worse.sum(dim=1)
                    ref_logps_better = ref_logps_better.sum(dim=1)
                    ref_logps_worse = ref_logps_worse.sum(dim=1)

                    pi_logps = torch.stack([pi_logps_better, pi_logps_worse], dim=1)
                    ref_logps = torch.stack([ref_logps_better, ref_logps_worse], dim=1)
                    losses, _ = self.dpo_loss(pi_logps, ref_logps) 
                    loss = losses.mean()
                    total_test_loss += loss.item()
            scheduler.step()
            print(f"Epoch [{epoch+1}/{epochs}], TEST Loss: {total_test_loss/len(test_dataloader)}")                    
                
    def training_loop(self, num_cycles=4, pop_size=100, ngen=7, cxpb=0.5, mutpb=0.2):
        random_numbers = [random.randint(0, 999) for _ in range(25)]
        for cycle in range(num_cycles):
            print(f"Cycle {cycle+1}/{num_cycles}")
#             random_numbers = [random.randint(0, 999) for _ in range(100)]
            seed_expr = generate_seed_expressions(random_numbers)
            pset = make_pset(self.num_vars)
            toolbox = setup_toolbox(pset, self.points)
            population, stats, hof = run_gp(toolbox, self.points, seed_expr,pset)
            print("best :- ",hof[0])
            preference_pairs = generate_preference_pairs(population, self.points,pset)
            self.train_transformer_dpo(preference_pairs)
            state_dict = self.model.state_dict()
            torch.save(state_dict, '/pscratch/sd/s/samyak09/model_weights.pth')
            self.beta = self.beta*5
            
        return self.model, population, stats, hof