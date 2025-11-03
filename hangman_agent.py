#!/usr/bin/env python3
"""
Hangman Agent - Balanced Aggressive Learning (FIXED & IMPROVED)
Features:
- Corrected Python syntax (__init__ not _init_)
- Faster learning with proper regularization
- Oracle-guided exploration (use bigram hints during training)
- Better reward shaping aligned with scoring formula
- Adaptive batch learning
- Portable file paths
- Better error handling
"""

import os
import sys
import random
import argparse
import json
import time
from pathlib import Path
from typing import Tuple, Set, Dict, List, Optional
import numpy as np
from collections import defaultdict, Counter, deque

# ============================================================================
# IMPORTS
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

# ============================================================================
# CONSTANTS
# ============================================================================

ALPHABET = [chr(i) for i in range(ord('a'), ord('z') + 1)]
ALPHA_IDX = {c: i for i, c in enumerate(ALPHABET)}
MAX_WORD_LEN = 15
MIN_WORD_LEN = 2
MAX_WRONG = 6

# Reward shaping constants (aligned with scoring formula)
REPEAT_PENALTY = -4.0  # Discourage repeated guesses
WRONG_PENALTY = -2.0   # Penalize wrong guesses
BASE_CORRECT_REWARD = 2.0
WIN_BONUS = 20.0
LOSS_PENALTY = -20.0

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

class HyperParams:
    """Training hyperparameters with adaptive adjustment."""
    
    def __init__(self):
        # Training
        self.episodes = 3000
        self.batch_size = 256
        self.learn_iterations = 3
        self.replay_size = 100000
        
        # Learning rates
        self.lr = 1e-3
        self.lr_min = 1e-5
        self.lr_max = 2e-3
        self.weight_decay = 5e-5
        
        # Exploration
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 0.997
        
        # Oracle-guided exploration
        self.oracle_guidance = 0.3
        self.oracle_decay = 0.9995
        
        # Network
        self.hidden_size = 256
        self.dropout = 0.25
        
        # RL
        self.gamma = 0.98
        self.tau = 0.01

    def adjust_from_performance(self, train_rate: float, eval_rate: float, 
                               episode: int) -> None:
        """Dynamically adjust hyperparameters based on performance."""
        gap = train_rate - eval_rate
        
        print(f"  [DEBUG] Train: {train_rate:.1%}, Eval: {eval_rate:.1%}, Gap: {gap:.1%}")
        
        # Overfitting detection
        if gap > 0.12:
            self.dropout = min(0.4, self.dropout + 0.05)
            self.weight_decay = min(1e-3, self.weight_decay * 1.3)
            self.learn_iterations = max(1, self.learn_iterations - 1)
            print(f"  [OVERFIT] Gap too large → Dropout={self.dropout:.2f}, Iter={self.learn_iterations}")
        
        # Performance-based adjustments
        if eval_rate < 0.08:  # Very poor
            self.lr = self.lr_max
            self.learn_iterations = 4
            self.oracle_guidance = 0.4
            print(f"  [BOOST] Eval {eval_rate:.1%} → LR={self.lr:.2e}, Oracle={self.oracle_guidance:.2f}")
            
        elif eval_rate < 0.15:  # Poor
            self.lr = 1.2e-3
            self.learn_iterations = 3
            self.oracle_guidance = 0.3
            print(f"  [RECOVER] Eval {eval_rate:.1%} → LR={self.lr:.2e}")
            
        elif eval_rate < 0.25:  # Moderate
            self.lr = 8e-4
            self.learn_iterations = 2
            self.oracle_guidance = 0.2
            print(f"  [IMPROVE] Eval {eval_rate:.1%} → LR={self.lr:.2e}")
            
        else:  # Good
            self.lr = max(self.lr_min, 5e-4)
            self.learn_iterations = 2
            self.oracle_guidance = 0.1
            print(f"  [STABLE] Eval {eval_rate:.1%} → LR={self.lr:.2e}")

# ============================================================================
# DATA & ORACLE
# ============================================================================

class BigramOracle:
    """Bigram-based probabilistic oracle for letter suggestions."""
    
    def __init__(self, words: List[str]):
        """
        Initialize oracle with word corpus.
        
        Args:
            words: List of training words
        """
        self.words = [w.lower() for w in words if w.isalpha()]
        self.alphabet = ALPHABET
        self.unigram: Counter = Counter()
        self.bigram: Dict = defaultdict(Counter)
        self.vocab_by_len: Dict = defaultdict(list)
        self._train()

    def _train(self) -> None:
        """Build unigram and bigram statistics from corpus."""
        for w in self.words:
            self.vocab_by_len[len(w)].append(w)
            prev = None
            for ch in w:
                self.unigram[ch] += 1
                if prev:
                    self.bigram[prev][ch] += 1
                prev = ch
        self.V = sum(self.unigram.values())
        print(f"  Oracle trained: {len(self.words)} words, V={self.V}")

    def get_probs(self, mask: str, guessed: Set[str]) -> np.ndarray:
        """
        Get probability distribution over remaining letters.
        
        Args:
            mask: Current game state (e.g., "a_c_e")
            guessed: Set of already guessed letters
            
        Returns:
            Probability vector of length 26
        """
        mask = mask.lower()
        L = len(mask)

        # Find candidate words matching pattern
        candidates = [w for w in self.vocab_by_len.get(L, [])
                     if all(m == '_' or m == w[i] for i, m in enumerate(mask))
                     and all(c not in guessed for i, c in enumerate(w) if mask[i] == '_')]

        probs = np.zeros(26, dtype=np.float32)

        if candidates:
            # Use candidate-based probabilities
            counts = Counter()
            for w in candidates:
                for i, ch in enumerate(w):
                    if mask[i] == '_' and ch not in guessed:
                        counts[ch] += 1

            if counts:
                total = sum(counts.values())
                for i, ch in enumerate(self.alphabet):
                    probs[i] = counts[ch] / total
                return probs

        # Fallback: Laplace-smoothed unigram
        total = self.V + 26
        for i, ch in enumerate(self.alphabet):
            if ch not in guessed:
                probs[i] = (self.unigram[ch] + 1) / total

        s = probs.sum()
        if s > 0:
            probs /= s

        return probs
    
    def suggest_letter(self, mask: str, guessed: Set[str]) -> int:
        """Get best letter index suggestion from oracle."""
        probs = self.get_probs(mask, guessed)
        avail = np.array([ALPHABET[i] not in guessed for i in range(26)])
        probs[~avail] = 0
        if probs.sum() > 0:
            return int(np.argmax(probs))
        return np.random.randint(26)

# ============================================================================
# ENVIRONMENT
# ============================================================================

class HangmanEnv:
    """Hangman game environment."""
    
    def __init__(self, word: str, max_wrong: int = MAX_WRONG):
        """
        Initialize game environment.
        
        Args:
            word: Secret word to guess
            max_wrong: Maximum wrong guesses allowed
        """
        self.word = word.lower()
        self.max_wrong = max_wrong
        self.reset()

    def reset(self) -> Tuple[str, set, int]:
        """Reset game state."""
        self.guessed: Set[str] = set()
        self.wrong = 0
        self.done = False
        self.revealed = list('_' * len(self.word))
        return self.get_state()

    def get_state(self) -> Tuple[str, Set[str], int]:
        """Get current game state."""
        return ''.join(self.revealed), set(self.guessed), self.max_wrong - self.wrong

    def step(self, letter: str) -> Tuple[Tuple[str, Set[str], int], float, bool]:
        """
        Execute one game step.
        
        Args:
            letter: Letter to guess
            
        Returns:
            (state, reward, done)
        """
        letter = letter.lower()
        reward = 0.0

        # Repeated guess
        if letter in self.guessed:
            reward = REPEAT_PENALTY
            return self.get_state(), reward, self.done

        self.guessed.add(letter)
        
        if letter in self.word:
            # Correct guess
            count = self.word.count(letter)
            for i, ch in enumerate(self.word):
                if ch == letter:
                    self.revealed[i] = letter
            
            # Reward scales with letter frequency and progress
            revealed_ratio = sum(1 for c in self.revealed if c != '_') / len(self.word)
            reward = BASE_CORRECT_REWARD * count + revealed_ratio * 2.0
        else:
            # Wrong guess
            self.wrong += 1
            reward = WRONG_PENALTY

        # Terminal state checks
        if '_' not in self.revealed:
            self.done = True
            reward += WIN_BONUS
        elif self.wrong >= self.max_wrong:
            self.done = True
            reward += LOSS_PENALTY

        return self.get_state(), reward, self.done

# ============================================================================
# NEURAL NETWORK
# ============================================================================

class DuelingDQN(nn.Module):
    """Dueling DQN architecture for letter selection."""
    
    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.25):
        """
        Initialize network.
        
        Args:
            input_dim: Input feature dimension
            hidden: Hidden layer size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 1)
        )
        
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, 26)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        f = self.feat(x)
        v = self.value(f)
        a = self.adv(f)
        return v + (a - a.mean(dim=1, keepdim=True))

# ============================================================================
# STATE ENCODING
# ============================================================================

def encode_state(mask: str, guessed: Set[str], lives: int, 
                hmm: np.ndarray, max_len: int = MAX_WORD_LEN) -> np.ndarray:
    """
    Encode game state into feature vector.
    
    Args:
        mask: Current masked word
        guessed: Set of guessed letters
        lives: Remaining lives
        hmm: HMM probability vector from oracle
        max_len: Maximum word length for padding
        
    Returns:
        Feature vector for neural network
    """
    # Position encoding with revealed letters
    pos_vec = np.zeros(max_len * 27, dtype=np.float32)
    for i in range(min(len(mask), max_len)):
        if mask[i] == '_':
            pos_vec[i * 27 + 26] = 1.0
        else:
            pos_vec[i * 27 + ALPHA_IDX[mask[i]]] = 1.0

    # Guessed letters binary vector
    guess_vec = np.zeros(26, dtype=np.float32)
    for c in guessed:
        if c in ALPHA_IDX:
            guess_vec[ALPHA_IDX[c]] = 1.0

    # Normalized features
    life_vec = np.array([lives / MAX_WRONG], dtype=np.float32)
    len_vec = np.array([len(mask) / max_len], dtype=np.float32)
    
    return np.concatenate([pos_vec, guess_vec, life_vec, len_vec, hmm])

# ============================================================================
# AGENT
# ============================================================================

class Agent:
    """Reinforcement learning agent for Hangman."""
    
    def __init__(self, input_dim: int, device: str, hp: HyperParams):
        """
        Initialize agent.
        
        Args:
            input_dim: Input feature dimension
            device: 'cpu' or 'cuda'
            hp: Hyperparameters
        """
        self.device = device
        self.hp = hp
        
        self.net = DuelingDQN(input_dim, hidden=hp.hidden_size, 
                             dropout=hp.dropout).to(device)
        self.target = DuelingDQN(input_dim, hidden=hp.hidden_size, 
                                dropout=hp.dropout).to(device)
        self.target.load_state_dict(self.net.state_dict())
        
        self.opt = optim.AdamW(self.net.parameters(), lr=hp.lr, 
                              weight_decay=hp.weight_decay)
        self.replay = deque(maxlen=hp.replay_size)
        self.priorities = deque(maxlen=hp.replay_size)
        self.updates = 0

    def act(self, state: np.ndarray, available_mask: np.ndarray, 
            eps: float, oracle_action: Optional[int] = None, 
            oracle_guidance: float = 0.0) -> int:
        """
        Select action (letter to guess).
        
        Args:
            state: Encoded state vector
            available_mask: Boolean mask of available actions
            eps: Exploration rate
            oracle_action: Suggested action from oracle
            oracle_guidance: Probability of following oracle
            
        Returns:
            Letter index (0-25)
        """
        # Oracle-guided exploration
        if oracle_action is not None and random.random() < oracle_guidance:
            return oracle_action
        
        # Epsilon-greedy exploration
        if random.random() < eps:
            valid_actions = [i for i in range(26) if available_mask[i]]
            return random.choice(valid_actions) if valid_actions else 0

        # Policy action
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, 
                           device=self.device).unsqueeze(0)
            q = self.net(x).cpu().numpy().flatten()
            q[~available_mask] = -1e9
            return int(np.argmax(q))

    def store(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, 
             d: bool, td_error: float = 1.0) -> None:
        """Store experience in replay buffer with priority."""
        self.replay.append((s, a, r, s2, d))
        self.priorities.append((abs(td_error) + 1e-6) ** 0.6)

    def soft_update(self) -> None:
        """Soft update target network."""
        for target_param, param in zip(self.target.parameters(), 
                                      self.net.parameters()):
            target_param.data.copy_(self.hp.tau * param.data + 
                                   (1.0 - self.hp.tau) * target_param.data)

    def update_learning_rate(self, new_lr: float) -> None:
        """Update optimizer learning rate."""
        for param_group in self.opt.param_groups:
            param_group['lr'] = new_lr

    def update_dropout(self, new_dropout: float) -> None:
        """Update dropout rates in both networks."""
        for module in self.net.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_dropout
        for module in self.target.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_dropout

    def learn(self) -> None:
        """Update agent using prioritized experience replay."""
        bs = self.hp.batch_size
        if len(self.replay) < bs:
            return

        # Prioritized sampling
        probs = np.array(self.priorities, dtype=np.float64)
        probs = probs / (probs.sum() + 1e-8)
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs / (probs.sum() + 1e-8)
        
        idx = np.random.choice(len(self.replay), bs, p=probs, replace=False)
        wts = (len(self.replay) * probs[idx]) ** (-0.4)
        wts = wts / (wts.max() + 1e-8)

        # Unpack batch
        batch = [self.replay[i] for i in idx]
        s = np.array([b[0] for b in batch], dtype=np.float32)
        a = np.array([b[1] for b in batch], dtype=np.int64)
        r = np.array([b[2] for b in batch], dtype=np.float32)
        s2 = np.array([b[3] for b in batch], dtype=np.float32)
        d = np.array([b[4] for b in batch], dtype=np.float32)

        # Convert to tensors
        s_t = torch.tensor(s, device=self.device, dtype=torch.float32)
        a_t = torch.tensor(a, device=self.device, dtype=torch.long)
        r_t = torch.tensor(r, device=self.device, dtype=torch.float32)
        s2_t = torch.tensor(s2, device=self.device, dtype=torch.float32)
        d_t = torch.tensor(d, device=self.device, dtype=torch.float32)
        w_t = torch.tensor(wts, device=self.device, dtype=torch.float32)

        # Compute Q-values
        self.net.train()
        q = self.net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.net(s2_t).argmax(1)
            q_next = self.target(s2_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            y = r_t + (1 - d_t) * self.hp.gamma * q_next

        # Weighted loss
        loss = (w_t * F.smooth_l1_loss(q, y, reduction='none')).mean()

        # Update priorities
        td_err = (q - y).detach().cpu().numpy()
        for i, ii in enumerate(idx):
            self.priorities[ii] = (abs(td_err[i]) + 1e-6) ** 0.6

        # Optimize
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        self.updates += 1
        if self.updates % 2 == 0:
            self.soft_update()

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def mini_eval(agent: Agent, oracle: BigramOracle, words: List[str], 
             device: str, games: int = 100) -> float:
    """Quick evaluation on subset of games."""
    wins = 0
    valid = [w for w in words if w.isalpha() and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN]
    
    if not valid:
        return 0.0
    
    for g in range(min(games, len(valid))):
        word = valid[g]
        env = HangmanEnv(word)
        state = env.reset()

        while not env.done:
            mask, guessed, lives = state
            hmm = oracle.get_probs(mask, guessed)
            s_vec = encode_state(mask, guessed, lives, hmm)
            avail = np.array([ALPHABET[i] not in guessed for i in range(26)])
            a = agent.act(s_vec, avail, eps=0.0)
            letter = ALPHABET[a]
            state, _, done = env.step(letter)

        if '_' not in state[0]:
            wins += 1
    
    return wins / games

def train(agent: Agent, oracle: BigramOracle, train_words: List[str], 
         test_words: List[str], hp: HyperParams, device: str = 'cpu') -> Dict:
    """Training loop with curriculum learning."""
    print(f"\nTraining {hp.episodes} episodes on {device.upper()}...")
    start = time.time()
    
    wins = losses = 0
    tp = fp = fn = 0
    window_wins = deque(maxlen=300)

    for ep in range(1, hp.episodes + 1):
        # Curriculum learning
        if ep < 800:
            valid_words = [w for w in train_words if w.isalpha() and 4 <= len(w) <= 8]
        elif ep < 1800:
            valid_words = [w for w in train_words if w.isalpha() and 3 <= len(w) <= 11]
        else:
            valid_words = [w for w in train_words if w.isalpha() and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN]
        
        if not valid_words:
            continue
        
        word = random.choice(valid_words)
        env = HangmanEnv(word)
        state = env.reset()
        
        unique_letters_in_word = set(word.lower())
        guesses_this_game: Set[str] = set()

        while not env.done:
            mask, guessed, lives = state
            hmm = oracle.get_probs(mask, guessed)
            s_vec = encode_state(mask, guessed, lives, hmm)

            avail = np.array([ALPHABET[i] not in guessed for i in range(26)])
            
            eps = max(hp.eps_end, hp.eps_start * (hp.eps_decay ** ep))
            oracle_guidance = hp.oracle_guidance * (hp.oracle_decay ** ep)
            oracle_action = oracle.suggest_letter(mask, guessed)
            
            a = agent.act(s_vec, avail, eps, oracle_action, oracle_guidance)
            letter = ALPHABET[a]

            state, r, done = env.step(letter)
            
            # Track metrics
            if letter not in guesses_this_game:
                guesses_this_game.add(letter)
                if letter in unique_letters_in_word:
                    tp += 1
                else:
                    fp += 1
            
            mask2, g2, l2 = state
            hmm2 = oracle.get_probs(mask2, g2)
            s2_vec = encode_state(mask2, g2, l2, hmm2)

            agent.store(s_vec, a, r, s2_vec, done)
            
            # Adaptive learning
            for _ in range(hp.learn_iterations):
                agent.learn()

        missed_letters = unique_letters_in_word - guesses_this_game
        fn += len(missed_letters)

        game_won = '_' not in state[0]
        if game_won:
            wins += 1
            window_wins.append(1)
        else:
            losses += 1
            window_wins.append(0)

        # Evaluation checkpoint
        if ep % 300 == 0:
            elapsed = time.time() - start
            train_win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            window_win_rate = sum(window_wins) / len(window_wins) if window_wins else 0
            
            eval_win_rate = mini_eval(agent, oracle, test_words, device, games=100)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  Ep {ep:4d} | WinRate {train_win_rate:.2%} | Eval {eval_win_rate:.2%} | "
                  f"F1 {f1:.3f} | Eps {eps:.3f} | Time {elapsed:.0f}s")
            
            hp.adjust_from_performance(window_win_rate, eval_win_rate, ep)
            agent.update_learning_rate(hp.lr)
            agent.update_dropout(hp.dropout)
            
            wins = losses = 0

    print(f"✓ Done in {time.time() - start:.0f}s")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }

def eval_agent(agent: Agent, oracle: BigramOracle, words: List[str], 
              device: str = 'cpu', games: int = 2000) -> Optional[Dict]:
    """Evaluate agent on test set."""
    print(f"\nEvaluating {games} games...")
    wins = wrongs = repeats = 0
    tp = fp = fn = 0

    valid = [w for w in words if w.isalpha() and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN]
    if not valid:
        print("ERROR: No valid words for evaluation")
        return None

    for g in range(games):
        word = valid[g % len(valid)]
        env = HangmanEnv(word)
        state = env.reset()
        
        unique_letters = set(word.lower())
        guesses_made: Set[str] = set()

        while not env.done:
            mask, guessed, lives = state
            hmm = oracle.get_probs(mask, guessed)
            s_vec = encode_state(mask, guessed, lives, hmm)

            avail = np.array([ALPHABET[i] not in guessed for i in range(26)])
            a = agent.act(s_vec, avail, eps=0.0)
            letter = ALPHABET[a]

            if letter in guessed:
                repeats += 1
            elif letter not in guesses_made:
                guesses_made.add(letter)
                if letter in unique_letters:
                    tp += 1
                else:
                    fp += 1
                    wrongs += 1

            state, _, done = env.step(letter)

        missed = unique_letters - guesses_made
        fn += len(missed)

        if '_' not in state[0]:
            wins += 1

        if (g + 1) % 400 == 0:
            print(f"  {g + 1}/{games} | Wins {wins} ({wins / (g + 1) * 100:.1f}%)")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    success_rate = wins / games
    final_score = wins - (wrongs * 5) - (repeats * 2)

    return {
        'wins': wins,
        'games': games,
        'success_rate': success_rate,
        'wrong_guesses': wrongs,
        'repeated_guesses': repeats,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'final_score': final_score
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Hangman Agent with DQN')
    parser.add_argument('--episodes', type=int, default=3000, help='Training episodes')
    parser.add_argument('--device', default='auto', choices=['auto', 'gpu', 'cpu'],
                       help='Compute device')
    parser.add_argument('--games', type=int, default=2000, help='Evaluation games')
    parser.add_argument('--corpus', type=str, default=r'C:\Users\karth\PROJECTS\mlhackathon\Hangman_ML_Hackathon\Data\corpus.txt', help='Path to corpus')
    parser.add_argument('--test', type=str, default=r'C:\Users\karth\PROJECTS\mlhackathon\Hangman_ML_Hackathon\Data\test.txt', help='Path to test set')
    parser.add_argument('--output', type=str, default='results.json', help='Output file')
    args = parser.parse_args()

    # Device selection
    device = 'cuda' if (args.device == 'auto' or args.device == 'gpu') and torch.cuda.is_available() else 'cpu'
    if args.device == 'cpu':
        device = 'cpu'

    print("=" * 70)
    print("HANGMAN AGENT - ORACLE-GUIDED BALANCED LEARNING")
    print("=" * 70)
    print(f"Device: {device.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Evaluation games: {args.games}")

    # Load data
    print("\nLoading data...")
    
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"ERROR: Corpus file '{args.corpus}' not found")
        sys.exit(1)
    
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_words = [line.strip() for line in f if line.strip()]

    test_path = Path(args.test)
    if test_path.exists():
        with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
            test_words = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: Test file '{args.test}' not found, using training set")
        test_words = train_words

    print(f"✓ Train: {len(train_words)} | Test: {len(test_words)}")

    # Build models
    print("\nBuilding models...")
    oracle = BigramOracle(train_words)
    input_dim = MAX_WORD_LEN * 27 + 26 + 1 + 1 + 26
    
    hp = HyperParams()
    hp.episodes = args.episodes
    
    agent = Agent(input_dim, device, hp)
    print(f"✓ Ready (input_dim={input_dim})")

    # Train
    train_metrics = train(agent, oracle, train_words, test_words, hp, device=device)

    # Save model
    torch.save({
        'model': agent.net.state_dict(),
        'optimizer': agent.opt.state_dict(),
        'input_dim': input_dim,
        'train_metrics': train_metrics
    }, 'model.pth')
    print("✓ Model saved to model.pth")

    # Evaluate
    results = eval_agent(agent, oracle, test_words, device=device, games=args.games)

    if results:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Wins: {results['wins']}/{results['games']}")
        print(f"Success Rate: {results['success_rate'] * 100:.2f}%")
        print(f"Wrong Guesses: {results['wrong_guesses']}")
        print(f"Repeated Guesses: {results['repeated_guesses']}")
        print()
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print()
        print(f"True Positives: {results['true_positives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"False Negatives: {results['false_negatives']}")
        print()
        print(f"Final Score: {results['final_score']:.0f}")
        print("=" * 70)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {args.output}")

if __name__ == '__main__':
    main()