import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split


def group_sequences_and_labels(df, features, token_column='mature_top1_token', end_token=["<|end_of_text|>"]):
    """
    Groups consecutive rows into sequences.
    Each row contains features (from `features`) and a token-level label.
    A row with the value end_token in token_column marks the end of a sequence.
    
    Returns:
        sequences: list of np.ndarray of shape (T, feature_dim)
        token_labels: list of np.ndarray of shape (T,) for token labels
    """
    sequences = []
    token_labels = []
    current_seq = []
    current_labels = []
    for _, row in df.iterrows():
        token_features = row[features].values.astype(np.float32)
        current_seq.append(token_features)
        current_labels.append(int(row['label']))  # label is assumed 0 or 1
        if row[token_column] in end_token:
            sequences.append(np.array(current_seq))
            token_labels.append(np.array(current_labels))
            current_seq = []
            current_labels = []
    if current_seq:
        sequences.append(np.array(current_seq))
        token_labels.append(np.array(current_labels))
    return sequences, token_labels


class DolaSeqTokenEnv(gym.Env):
    """
    An environment where the agent makes a decision at every token.
    The episode ends when the token (at token_column_index) equals end_token or when sequence ends.
    The reward is computed over the entire sequence using a predefined reward scheme.
    """
    def __init__(self, sequences, token_labels, balanced_sampling: bool = False,
                 token_column_index=0, end_token="<|end_of_text|>"):
        super().__init__()
        self.sequences = sequences
        self.token_labels = token_labels
        self.n_samples = len(sequences)
        # Reward scheme (penalize false negatives heavily)
        self.reward_tp = 1.0
        self.reward_fn = -4.0
        self.reward_fp = -1.0
        self.reward_tn = 1.0
        self.balanced_sampling = balanced_sampling
        self.token_column_index = token_column_index
        self.end_token = end_token
        
        obs_dim = sequences[0].shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.current_index = None
        self.current_seq = None
        self.current_labels = None
        self.current_step = None
        self.predictions = []
        
        if self.balanced_sampling:
            # Use first token label as proxy for sequence label
            self.seq_labels = np.array([labels[0] for labels in token_labels])
            self.class_indices = {
                0: np.where(self.seq_labels == 0)[0],
                1: np.where(self.seq_labels == 1)[0]
            }
    
    def reset(self):
        if self.balanced_sampling:
            chosen_label = np.random.choice([0, 1])
            indices = self.class_indices[chosen_label]
            self.current_index = np.random.choice(indices)
        else:
            self.current_index = np.random.randint(0, self.n_samples)
        self.current_seq = self.sequences[self.current_index]
        self.current_labels = self.token_labels[self.current_index]
        self.current_step = 0
        self.predictions = []
        return self.current_seq[self.current_step]
    
    def step(self, action: int):
        self.predictions.append(action)
        token_value = self.current_seq[self.current_step][self.token_column_index]
        # Check terminal condition: either explicit end token or end of sequence.
        is_terminal = (str(token_value) == self.end_token) or (self.current_step == len(self.current_seq) - 1)
        if is_terminal:
            # Compute cumulative reward over the sequence
            total_reward = 0.0
            for pred, true_label in zip(self.predictions, self.current_labels):
                if true_label == 1 and pred == 1:
                    total_reward += self.reward_tp
                elif true_label == 1 and pred == 0:
                    total_reward += self.reward_fn
                elif true_label == 0 and pred == 1:
                    total_reward += self.reward_fp
                else:
                    total_reward += self.reward_tn
            done = True
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"sequence_length": len(self.current_seq), "predictions": self.predictions}
            return next_obs, total_reward, done, info
        else:
            self.current_step += 1
            obs = self.current_seq[self.current_step]
            done = False
            reward = 0.0
            info = {"step": self.current_step}
            return obs, reward, done, info

def evaluate_policy(agent, env, n_samples: int):
    """
    Evaluate the BCQ agent on n_samples sequences.
    Aggregates token-level predictions to compute standard metrics.
    """
    all_predictions = []
    all_true_labels = []
    
    for _ in range(n_samples):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
        all_predictions.extend(env.predictions)
        all_true_labels.extend(env.current_labels[:len(env.predictions)])
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    tp = np.sum((all_true_labels == 1) & (all_predictions == 1))
    fp = np.sum((all_true_labels == 0) & (all_predictions == 1))
    tn = np.sum((all_true_labels == 0) & (all_predictions == 0))
    fn = np.sum((all_true_labels == 1) & (all_predictions == 0))
    
    print(f"Total Tokens: {len(all_true_labels)}")
    print(f"Ground Truth: Positives={np.sum(all_true_labels==1)}, Negatives={np.sum(all_true_labels==0)}")
    print(f"Predictions: Positives={np.sum(all_predictions==1)}, Negatives={np.sum(all_predictions==0)}")
    
    return {"fn/(fn+tp)": fn/(tp+fn+1e-9), "tn/(tn+fp)": tn/(tn+fp+1e-9),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


class TransitionDataset(Dataset):
    """
    Offline dataset of transitions collected from your sequences.
    Each sample is a tuple: (state, action, reward, next_state, done)
    For episodes (sequences), intermediate tokens have reward 0; only terminal tokens have a nonzero reward.
    """
    def __init__(self, sequences, token_labels, obs_shape):
        self.transitions = []
        for seq, labels in zip(sequences, token_labels):
            T = len(seq)
            # For each step in the sequence
            for t in range(T):
                state = seq[t]
                action = int(labels[t])  # assume behavior policy was optimal (ground truth)
                # For nonterminal steps, reward is 0.
                if t < T - 1:
                    reward = 0.0
                    next_state = seq[t+1]
                    done = 0
                else:
                    # Terminal reward: sum over the sequence computed using the reward scheme.
                    total_reward = 0.0
                    for pred, true_label in zip(labels, labels):
                        # Using the reward scheme (assuming optimal actions)
                        if true_label == 1:
                            total_reward += 1.0  # reward_tp
                        else:
                            total_reward += 1.4  # reward_tn
                    reward = total_reward
                    next_state = np.zeros(obs_shape, dtype=np.float32)
                    done = 1
                self.transitions.append((state, action, reward, next_state, done))
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.transitions[idx]
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))


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
    
    def pretrain_bc(self, dataloader, epochs=10):
        """Pre-train the behavior cloning network on the offline dataset."""
        self.bc_net.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for state, action, _, _, _ in dataloader:
                state = state.to(self.device)
                action = action.to(self.device)
                logits = self.bc_net(state)
                loss = self.ce_loss(logits, action)
                self.bc_optimizer.zero_grad()
                loss.backward()
                self.bc_optimizer.step()
                total_loss += loss.item()
            print(f"BC Epoch {epoch+1}/{epochs} loss: {total_loss/len(dataloader):.4f}")
        self.bc_net.eval()
    
    def train_q(self, dataloader, epochs=10, update_target_every=100):
        """Train the Q network using offline transitions with BCQ constraints."""
        self.q_net.train()
        total_steps = 0
        for epoch in range(epochs):
            total_loss = 0.0
            for state, action, reward, next_state, done in dataloader:
                state = state.to(self.device)
                action = action.to(self.device)
                reward = reward.to(self.device)
                next_state = next_state.to(self.device)
                done = done.to(self.device)
                
                # Q value for current state-action pair
                q_val = self.q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    # Get behavior probabilities for next_state using bc_net.
                    bc_logits = self.bc_net(next_state)
                    bc_probs = torch.softmax(bc_logits, dim=1)
                    # Allowed actions: those with probability above threshold.
                    allowed_mask = (bc_probs > self.bc_threshold).float()
                    # If no action passes the threshold, allow all actions.
                    no_allowed = (allowed_mask.sum(dim=1) == 0).unsqueeze(1)
                    allowed_mask = allowed_mask + no_allowed
                    
                    # Q_target for next_state
                    q_next = self.q_target(next_state)
                    # Mask out actions not allowed by behavior model.
                    q_next_masked = q_next - (1 - allowed_mask) * 1e8
                    max_q_next, _ = torch.max(q_next_masked, dim=1)
                    target = reward + self.gamma * max_q_next * (1 - done)
                
                loss = nn.MSELoss()(q_val, target)
                self.q_optimizer.zero_grad()
                loss.backward()
                self.q_optimizer.step()
                
                total_loss += loss.item()
                total_steps += 1
                if total_steps % update_target_every == 0:
                    self.q_target.load_state_dict(self.q_net.state_dict())
            print(f"Q Training Epoch {epoch+1}/{epochs} loss: {total_loss/len(dataloader):.4f}")
        self.q_net.eval()
    
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


if __name__ == '__main__':
    df_train = pd.read_csv('/home/hxxzhang/DoLa/train_classifier/llama-strqa-low.csv')
    
    features = [col for col in df_train.columns if 'token_id' in col or 'prob' in col]
    token_column = 'mature_top1_token'
    end_token = ["<|end_of_text|>","Q"]
    
    sequences, token_labels = group_sequences_and_labels(df_train, features, token_column=token_column, end_token=end_token)
    
    # Split into training and testing sequences.
    seq_train, seq_test, labels_train, labels_test = train_test_split(sequences, token_labels, test_size=0.2, random_state=42)
    obs_shape = seq_train[0].shape[1]
    offline_dataset = TransitionDataset(seq_train, labels_train, obs_shape)
    dataloader = DataLoader(offline_dataset, batch_size=1024, shuffle=True)
    
    state_dim = obs_shape
    action_dim = 2  # binary actions: 0 and 1
    device = 'cuda:6'
    agent = BCQAgent(state_dim, action_dim, hidden_layers=[1024, 1024, 512, 512, 256, 256], gamma=1.0, lr=3e-4, device=device, bc_threshold=0.3)
    print("Pretraining behavior cloning network ...")
    agent.pretrain_bc(dataloader, epochs=35)
    
    print("Training Q network with BCQ loss ...")
    agent.train_q(dataloader, epochs=130, update_target_every=200)
    
    model_save_path = "PATH TO MODEL"
    torch.save({
        'input_dim': agent.state_dim,
        'hidden_layers': agent.hidden_layers,
        'bc_threshold': agent.bc_threshold,
        'q_net': agent.q_net.state_dict(),
        'bc_net': agent.bc_net.state_dict(),
        'q_target': agent.q_target.state_dict(),
    }, model_save_path)