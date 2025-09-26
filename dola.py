import argparse
import time
import csv
import tqdm
import os
import json
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

class BCQAgent:
    def __init__(self, state_dim, action_dim, hidden_layers=[1024, 512, 256],
                 gamma=1.0, lr=1e-4, device='cpu', bc_threshold=0.3):
        self.device = device
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.bc_threshold = bc_threshold  # probability threshold for behavior constraint
        
        # Q network and target network
        self.q_net = self.build_network(state_dim, action_dim, hidden_layers).to(self.device)
        self.q_target = self.build_network(state_dim, action_dim, hidden_layers).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        
        # Behavior cloning network (trained via supervised learning)
        self.bc_net = self.build_network(state_dim, action_dim, hidden_layers).to(self.device)
        self.bc_optimizer = optim.AdamW(self.bc_net.parameters(), lr=lr)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def build_network(self, input_dim, output_dim, hidden_layers):
        layers = []
        last_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)
    
    def select_action(self, state):
        """Select action using the behavior constraint and Q network.
           state: a NumPy array (observation)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            bc_logits = self.bc_net(state_tensor)
            bc_probs = torch.softmax(bc_logits, dim=1).squeeze(0)
            q_vals = self.q_net(state_tensor).squeeze(0)
        
        # Get allowed actions from behavior model.
        allowed = (bc_probs > self.bc_threshold).nonzero(as_tuple=True)[0]
        if len(allowed) == 0:
            # If none are allowed, fall back to the behavior's highest probability action.
            action = torch.argmax(bc_probs).item()
        else:
            # Choose among allowed actions the one with highest Q-value.
            q_allowed = q_vals[allowed]
            best_idx = torch.argmax(q_allowed).item()
            action = allowed[best_idx].item()
        return action
    def BC_predict(self, state, deterministic=True):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            bc_logits = self.bc_net(state_tensor)
            bc_probs = torch.softmax(bc_logits, dim=1).squeeze(0)
        # Return the action with highest probability
        action = torch.argmax(bc_probs).item()
        # else:
        #     # Sample from the probability distribution
        #     action = torch.multinomial(bc_probs, 1).item()
        return action

class DoLa:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27, classifier_path='-1'):
        self.model_name = model_name
        self.device = device
        self.num_gpus = int(num_gpus)
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory
        self.classifier_path = classifier_path

        self.model, self.tokenizer, self.classifier = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}

        kwargs = {"device_map": self.device, "quantization_config": BitsAndBytesConfig(load_in_8bit=True,)} 
        if model_name =='deepseek-ai/DeepSeek-V2-Lite-Chat' or model_name =='mistralai/Codestral-22B-v0.1' or model_name =='google/gemma-3-12b-it':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,torch_dtype=torch.float16)
        elif model_name !='huggyllama/llama-7b':
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
            model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)

        # if 'cuda' in self.device and self.num_gpus == 1:
        #     # model.cuda()
        #     model.to(torch.device(self.device))
        #     print('loading model to',self.device)
        loaded_model = None
        if os.path.exists(self.classifier_path):
            print('Loading BCQ model ',self.classifier_path)
            checkpoint = torch.load(self.classifier_path, map_location=self.device)
            input_dim, hidden_layers = checkpoint['input_dim'],checkpoint['hidden_layers']
            loaded_model = BCQAgent(input_dim, 2, hidden_layers=hidden_layers, gamma=1.0, lr=1e-4, device=self.device, bc_threshold=0.3)
            loaded_model.q_net.load_state_dict(checkpoint['q_net'])
            loaded_model.q_target.load_state_dict(checkpoint['q_target'])
            loaded_model.bc_net.load_state_dict(checkpoint['bc_net'])
            # Set networks to evaluation mode
            loaded_model.q_net.eval()
            loaded_model.bc_net.eval()
        return model, tokenizer, loaded_model

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, **kwargs):
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'baseline':
                # outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                #                     output_scores=True, return_dict_in_generate=True, dola_decoding=False,
                #                     top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
                # Greedy search
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, do_sample=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == 'dola-static':
                assert mature_layer is not None, "mature_layer must be specified"
                assert premature_layer is not None, "premature_layer must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                                    mature_layer=mature_layer, premature_layer=premature_layer,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, **kwargs)
            elif mode in ['dola','actlcd']:
                # tranformers 4.46.3
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, do_sample=False,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, classifier=self.classifier, **kwargs)
                premature_layer_dist = None #outputs.premature_layer_dist
            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            # if verbose:
            #     print('MODEL OUTPUT: \n{0}'.format(output_str),"\nMODEL OUTPUT END")

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dola' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh