"""
hangman_agent.py

Single-file starter implementation for the Hangman HMM + RL challenge.

Includes:
- Corpus loader (expects Data/corpus[.txt]) and Data/test[.txt]
- HMM-like oracle using bigram + mask filtering
- Hangman environment (gym-like API)
- DQN Agent (PyTorch) using oracle probabilities as part of the state
- Integrated training + evaluation routine

When you run this file directly (`python hangman_agent.py`), it will:
1. Train the agent on the corpus (default 2000 episodes)
2. Save the model to `hangman_dqn.pth`
3. Evaluate on the test set
4. Save results to `results.json`

You can still run manually with flags:
- Train only:   python hangman_agent.py --train --episodes 3000
- Evaluate only: python hangman_agent.py --eval --model hangman_dqn.pth

"""

import os
import random
import argparse
import json
from collections import defaultdict, Counter
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise ImportError("This script requires PyTorch. Install with `pip install torch`.")

# --------------------- Corpus & HMM (n-gram oracle) ---------------------
class BigramOracle:
    def __init__(self, words):
        self.words = [w.lower() for w in words if w.isalpha()]
        self.alphabet = [chr(i) for i in range(ord('a'), ord('z')+1)]
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.vocab_by_len = defaultdict(list)
        self._train()

    def _train(self):
        for w in self.words:
            self.vocab_by_len[len(w)].append(w)
            prev = None
            for ch in w:
                self.unigram[ch] += 1
                if prev is not None:
                    self.bigram[prev][ch] += 1
                prev = ch
        self.V = sum(self.unigram.values())

    def candidate_letter_probs(self, mask, guessed):
        mask = mask.lower()
        L = len(mask)
        candidates = []
        for w in self.vocab_by_len.get(L, []):
            ok = True
            for mc, wc in zip(mask, w):
                if mc == '_':
                    if wc in guessed:
                        ok = False
                        break
                else:
                    if mc != wc:
                        ok = False
                        break
            if ok:
                candidates.append(w)
        counts = Counter()
        if candidates:
            for w in candidates:
                for i, ch in enumerate(w):
                    if mask[i] == '_' and ch not in guessed:
                        counts[ch] += 1
            total = sum(counts.values())
            if total == 0:
                return self._unigram_probs(guessed)
            probs = np.zeros(26, dtype=np.float32)
            for i, ch in enumerate(self.alphabet):
                probs[i] = counts[ch] / total
            return probs
        else:
            return self._unigram_probs(guessed)

    def _unigram_probs(self, guessed):
        probs = np.zeros(26, dtype=np.float32)
        total = sum(self.unigram.values()) + 26
        for i, ch in enumerate(self.alphabet):
            if ch in guessed:
                probs[i] = 0.0
            else:
                probs[i] = (self.unigram[ch] + 1) / total
        s = probs.sum()
        if s > 0:
            probs /= s
        return probs

# --------------------- Hangman Environment ---------------------
class HangmanEnv:
    def __init__(self, word, max_wrong=6):
        self.word = word.lower()
        self.max_wrong = max_wrong
        self.reset()

    def reset(self):
        self.guessed = set()
        self.wrong = 0
        self.done = False
        self.revealed = ['_' for _ in self.word]
        return self._get_obs()

    def _get_obs(self):
        return ''.join(self.revealed), set(self.guessed), self.max_wrong - self.wrong

    def step(self, letter):
        letter = letter.lower()
        if self.done:
            raise RuntimeError('step() called on finished env')
        reward = 0.0
        repeated = 0
        if letter in self.guessed:
            repeated = 1
            reward -= 0.5
        else:
            self.guessed.add(letter)
            if letter in self.word:
                for i, ch in enumerate(self.word):
                    if ch == letter:
                        self.revealed[i] = letter
                reward += 1.0
            else:
                self.wrong += 1
                reward -= 1.0
        if '_' not in self.revealed:
            self.done = True
            reward += 10.0
        elif self.wrong >= self.max_wrong:
            self.done = True
            reward -= 5.0
        return self._get_obs(), reward, self.done, {'repeated': repeated}

# --------------------- DQN Agent ---------------------
class DQN(nn.Module):
    def __init__(self, input_dim, hidden=128, output_dim=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, input_dim, device='cpu'):
        self.device = device
        self.net = DQN(input_dim).to(self.device)
        self.target = DQN(input_dim).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = optim.Adam(self.net.parameters(), lr=1e-3)
        self.replay = []
        self.max_replay = 10000

    def select(self, state_vec, avail_mask, eps=0.1):
        if random.random() < eps:
            choices = [i for i, v in enumerate(avail_mask) if v]
            return random.choice(choices)
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.net(x).cpu().numpy().flatten()
            q_masked = np.full_like(q, -1e9)
            for i in range(26):
                if avail_mask[i]:
                    q_masked[i] = q[i]
            return int(np.argmax(q_masked))

    def store(self, *transition):
        self.replay.append(transition)
        if len(self.replay) > self.max_replay:
            self.replay.pop(0)

    def update(self, batch_size=64, gamma=0.99):
        if len(self.replay) < batch_size:
            return
        batch = random.sample(self.replay, batch_size)
        s_b = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        a_b = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        r_b = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        s2_b = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        done_b = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        q = self.net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_target = self.target(s2_b).max(1)[0]
            y = r_b + (1.0 - done_b) * gamma * q_target
        loss = nn.functional.mse_loss(q, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def sync(self):
        self.target.load_state_dict(self.net.state_dict())

# --------------------- Utilities ---------------------
ALPH = [chr(i) for i in range(ord('a'), ord('z')+1)]
LETTER2IDX = {c: i for i, c in enumerate(ALPH)}

def encode_state(mask, guessed_set, lives_left, hmm_probs, max_len=15):
    mask = mask.lower()
    L = len(mask)
    pos_vec = np.zeros(max_len * 27, dtype=np.float32)
    for i in range(min(L, max_len)):
        ch = mask[i]
        if ch == '_':
            pos_vec[i * 27 + 26] = 1.0
        else:
            pos_vec[i * 27 + LETTER2IDX[ch]] = 1.0
    guessed_vec = np.zeros(26, dtype=np.float32)
    for c in guessed_set:
        if c in LETTER2IDX:
            guessed_vec[LETTER2IDX[c]] = 1.0
    lives_vec = np.array([lives_left / 6.0], dtype=np.float32)
    st = np.concatenate([pos_vec, guessed_vec, lives_vec, hmm_probs.astype(np.float32)])
    return st

# --------------------- Training and Evaluation ---------------------

def load_corpus(path):
    if os.path.isdir(path):
        fpath = None
        for candidate in os.listdir(path):
            if candidate.lower().startswith('corpus') and candidate.lower().endswith('.txt'):
                fpath = os.path.join(path, candidate)
                break
        if fpath is None:
            raise FileNotFoundError('No corpus file in directory')
    else:
        fpath = path
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        words = [line.strip() for line in f if line.strip()]
    return words

def train(agent, oracle, train_words, episodes=2000, max_len=15):
    target_sync_every = 200
    eps_start, eps_end = 0.5, 0.05
    
    print(f"Training for {episodes} episodes...")
    print(f"Epsilon decay: {eps_start} -> {eps_end}")
    print(f"Target network sync every {target_sync_every} episodes\n")
    
    for ep in range(1, episodes + 1):
        word = random.choice(train_words)
        if not word.isalpha() or len(word) > max_len:
            continue
        env = HangmanEnv(word)
        obs = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done:
            mask, guessed, lives = obs
            hmm_probs = oracle.candidate_letter_probs(mask, guessed)
            state_vec = encode_state(mask, guessed, lives, hmm_probs, max_len=max_len)
            avail_mask = [c not in guessed for c in ALPH]
            eps = max(eps_end, eps_start * (1 - ep / episodes))
            a_idx = agent.select(state_vec, avail_mask, eps=eps)
            letter = ALPH[a_idx]
            (mask2, guessed2, lives2), reward, done, info = env.step(letter)
            hmm2 = oracle.candidate_letter_probs(mask2, guessed2)
            state2_vec = encode_state(mask2, guessed2, lives2, hmm2, max_len=max_len)
            agent.store(state_vec, a_idx, reward, state2_vec, float(done))
            loss = agent.update()
            obs = (mask2, guessed2, lives2)
            total_r += reward
            steps += 1
        if ep % target_sync_every == 0:
            agent.sync()
        if ep % 100 == 0:
            print(f"Episode {ep}/{episodes} | Reward: {total_r:.2f} | Steps: {steps} | Epsilon: {eps:.3f} | Replay: {len(agent.replay)}")
    
    print(f"\nTraining complete! Replay buffer size: {len(agent.replay)}")
    return agent

def evaluate(agent, oracle, test_words, games=2000, max_len=15, verbose=True):
    wins = 0
    wrongs = 0
    repeats = 0
    games_played = 0
    
    # Filter valid test words upfront
    valid_words = [w for w in test_words if w.isalpha() and len(w) <= max_len]
    if len(valid_words) < games:
        print(f"Warning: Only {len(valid_words)} valid test words available, will cycle through them")
    
    for i in range(games):
        # Cycle through valid words if we don't have enough
        word = valid_words[i % len(valid_words)]
        
        env = HangmanEnv(word)
        obs = env.reset()
        done = False
        game_wrongs = 0
        game_repeats = 0
        
        while not done:
            mask, guessed, lives = obs
            hmm_probs = oracle.candidate_letter_probs(mask, guessed)
            state_vec = encode_state(mask, guessed, lives, hmm_probs, max_len=max_len)
            avail_mask = [c not in guessed for c in ALPH]
            a_idx = agent.select(state_vec, avail_mask, eps=0.0)
            letter = ALPH[a_idx]
            
            # Track wrong guesses before the step
            was_guessed = letter in guessed
            
            (mask, guessed, lives), reward, done, info = env.step(letter)
            
            if info.get('repeated', 0):
                game_repeats += 1
                repeats += 1
            elif letter not in word:
                # Only count as wrong if it wasn't repeated and not in word
                game_wrongs += 1
                wrongs += 1
        
        if '_' not in mask:
            wins += 1
        
        games_played += 1
        
        # Progress reporting
        if verbose and (i + 1) % 200 == 0:
            print(f"Progress: {i+1}/{games} games - Wins: {wins}, Wrongs: {wrongs}, Repeats: {repeats}")
    
    return {'wins': wins, 'games': games_played, 'wrongs': wrongs, 'repeats': repeats}

# --------------------- Main ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Data', help='path to Data directory or corpus file')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--model', default='hangman_dqn.pth')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    candidates = [os.path.join(args.data, 'corpus'), os.path.join(args.data, 'corpus.txt'), os.path.join(args.data, 'corpus.csv'), args.data]
    words = None
    for c in candidates:
        if os.path.exists(c):
            try:
                words = load_corpus(c)
                break
            except Exception:
                continue
    if words is None:
        raise FileNotFoundError('Could not find corpus in Data folder.')

    test_path = os.path.join(args.data, 'test')
    test_path_txt = os.path.join(args.data, 'test.txt')
    if os.path.exists(test_path):
        test_words = load_corpus(test_path)
    elif os.path.exists(test_path_txt):
        test_words = load_corpus(test_path_txt)
    else:
        random.shuffle(words)
        split = int(0.8 * len(words))
        test_words = words[split:split + 2000]
    train_words = words

    oracle = BigramOracle(train_words)
    max_len = 15
    input_dim = max_len * 27 + 26 + 1 + 26
    agent = Agent(input_dim, device=args.device)

    # Auto-integrated: train + eval
    print('='*60)
    print('Starting integrated train + eval pipeline...')
    print(f'Training corpus: {len(train_words)} words')
    print(f'Test set: {len(test_words)} words')
    print('='*60)
    
    print('\n--- TRAINING PHASE ---')
    train(agent, oracle, train_words, episodes=args.episodes, max_len=max_len)
    
    # Optionally save model
    if args.model:
        torch.save(agent.net.state_dict(), args.model)
        print(f'\nModel saved to {args.model}')
    
    print('\n--- EVALUATION PHASE ---')
    print('Evaluating on 2000 games from test set...\n')
    res = evaluate(agent, oracle, test_words, games=2000, max_len=max_len, verbose=True)
    
    # Calculate final score using the provided formula
    success_rate = res['wins'] / res['games']
    final_score = (success_rate * 2000) - (res['wrongs'] * 5) - (res['repeats'] * 2)
    
    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)
    print(f"Games Played:       {res['games']}")
    print(f"Wins:               {res['wins']}")
    print(f"Losses:             {res['games'] - res['wins']}")
    print(f"Success Rate:       {success_rate*100:.2f}%")
    print(f"Total Wrong Guesses: {res['wrongs']}")
    print(f"Total Repeated Guesses: {res['repeats']}")
    print(f"\nFINAL SCORE: {final_score:.2f}")
    print('='*60)
    
    out = {
        'results': res,
        'success_rate': success_rate,
        'final_score': final_score
    }
    with open('results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nDetailed results saved to results.json')

if __name__ == '__main__':
    main()
