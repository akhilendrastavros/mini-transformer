
import torch
from torch.nn import functional as F
import torch.nn as nn


"""

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
"""
batch_size = 16 #64
block_size = 8 #256
max_iters = 1500 #5000
eval_interval = 150 #500
learning_rate = 1e-2 #3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 #384 #386/6 means that every head is 64 dimmensional
n_head = 6
n_layer = 3 #6
dropout = 0.2
#-------------------------------------


torch.manual_seed(1337)


#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding = 'utf-8') as f:
  text = f.read()

print("length of dataset in characters: ", len(text))

# let's look at the first 1000 characters
print(text[:1000])

#keep track of all unique characters
#gets a sorted list of all the unique characters in the text
chars = sorted(list(set(text)))
#possible elements of our sequences
vocab_size = len(chars)
print(' '.join(chars))
print(vocab_size)


#create a mapping from chars to integers
stoi = {ch:i for i,ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #take a string, output list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #take list of integers, output string


"""Google uses SentencePiece, which encodes strings and text into integers. Read more on later. OpenAI uses tiktoken."""

#we can load in the entire text dataset and then load it into a tensor
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

"""We are never going to feed all the data to the transformer all at once. We sample random chunks of the dataset and train with chunks at a time. Each chunk has a blocksize (maximum length of chunk/set etc)."""

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):
    def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias = False)
      self.query = nn.Linear(n_embd, head_size, bias = False)
      self.value = nn.Linear(n_embd, head_size, bias = False)
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
      self.dropout = nn.Dropout(dropout)

    def forward(self, x):
      B,T,C = x.shape
      k = self.key(x) #(B,T,C)
      q = self.query(x) #(B,T,C)
      # compute attention scores ("affinities")
      wei = q @ k.transpose(-2, -1) * C**-0.5 #(B,T,C) @ (B,T,C) --> (B,T,T)
      wei = wei.masked_fill(self.tril[:T, :T] ==0, float('-inf')) # (B,T,T)
      wei = F.softmax(wei, dim = -1) # (B,T,T)
      wei = self.dropout(wei)
      #perform the weighted aggregation of the values
      v = self.value(x)
      out = wei @ v
      return out 
    

class MultiHeadAttention(nn.Module):
  "multiple heads of self-attention in parallel"

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
   out = torch.cat([h(x) for h in self.heads], dim = -1)
   out = self.dropout(self.proj(out))
   return out 


class FeedForward(nn.Module):
  "a simple linear layer followed by a non linearity"

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module):
  """Transformer Block: communication followed by computation"""

  def __init__(self, n_embd, n_head):
    #n_embd: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)    
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


#super simple bigram model
class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    #each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head =n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) #final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    #idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) #(B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T,C)
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x= self.ln_f(x)
    logits = self.lm_head(x) #(B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:,-1,:]
      probs = F.softmax(logits, dim =-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx,idx_next), dim = 1)
    return idx  
  

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
  print(f"Iteration: {iter}")
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step{iter}, trainloss{losses['train']:.4f}, val loss{losses['val']:.4f}")

  xb, yb = get_batch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1,1), dtype = torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
