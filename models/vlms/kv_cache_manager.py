import torch
import torch.nn as nn
class SequentialRNNGate(nn.Module):
    """
    A tiny LSTM-based gate that scores a chunk of tokens to decide whether we
    should KEEP (compress-aggregate) or DROP that chunk from the long-term KV cache.

    Shapes:
      - chunk: (B=1, C, D) where C is chunk_size, D is key/value dimensionality.
      - h_prev: (h, c) LSTM states; each is (num_layers=1, B=1, H_hidden)

    Returns:
      - gate_score: scalar in [0, 1]; >= threshold means "keep" (compress)
      - h_next: updated LSTM hidden state tuple
    """
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, chunk, h_prev):
        output, h_next = self.rnn(chunk, h_prev)
        last_out = output[:, -1, :]
        gate_score = torch.sigmoid(self.fc(last_out)).squeeze()
        return gate_score, h_next

def aggregate_mean(chunk):
    """
    Average-pool a chunk of tokens along time.
    Accepts (C, D) or (B, C, D) where B=1 for our use.
    Returns shape (1, D) so it can be inserted back as a single "compressed" token.
    """
    if chunk.dim() == 2:# (C, D)
        return chunk.mean(dim=0, keepdim=True)# (1, D)
    else:# (1, C, D)
        return chunk.mean(dim=1, keepdim=True)# (1, D)

class KVCacheManager:
    """
    Maintains an efficient KV cache with three regions:
      1) Far past: periodically compacted into fewer "summary" tokens.
      2) Middle: just beyond the recent window; processed in fixed-size chunks (C).
      3) Recent window (size W): always kept verbatim (no compression).

    When the running buffer reaches C and the cache length exceeds W:
      - Compute gate score with an RNN over that chunk.
      - If gate >= threshold -> keep the chunk but replace the oldest C tokens
        *beyond the window* with a single aggregated token (compress).
      - Else -> drop those C tokens entirely (forget).

    Parameters
    ----------
    window_size (W): int
        Number of most-recent tokens to keep verbatim.
    chunk_size (C): int
        How many tokens to consider per gate decision.
    aggregate_fn: Callable[[Tensor], Tensor]
        Aggregation function mapping (1, C, D) or (C, D) -> (1, D).
    rnn_gate: nn.Module
        Produces a gate score in [0, 1] for a chunk.
    threshold: float
        Decision threshold for keeping/compressing vs dropping.
    """
    def __init__(self, window_size, chunk_size, aggregate_fn, rnn_gate, threshold):
        self.W = window_size
        self.C = chunk_size
        self.aggregate = aggregate_fn
        self.rnn_gate = rnn_gate
        self.threshold = threshold

    def compress_prefill_kv(self, seq_KV, h_state):
        """
        Optional prefill path to pre-compress a long sequence before decoding.

        Input:
          - seq_KV: (N, D) sequence of keys or values
          - h_state: LSTM state

        Output:
          - compressed: (<=N, D) compacted sequence
          - h_state: updated LSTM state
        """
        N, D = seq_KV.shape
        retained = []

        # Process everything except the last W tokens in chunks of size C
        for i in range(0, N - self.W, self.C):
            chunk = seq_KV[i : i + self.C] # (C, D)
            agg_chunk = self.aggregate(chunk) # (1, D)
            gate_score, h_state = self.rnn_gate(chunk.unsqueeze(0), h_state) # gate over (1, C, D)
            if gate_score >= self.threshold:
                 # Keep a compressed token
                retained.append(agg_chunk.squeeze(0))# (D,)

        # Always keep the recent window verbatim
        for i in range(N - self.W, N):
            retained.append(seq_KV[i])
        return torch.stack(retained, dim=0), h_state

    def online_update(self, KV_cache,v_cache, new_kv,new_v, h_state, buffer=None,buffer_v=None):
        """
        Incremental update per new token during decoding.

        Inputs:
          - KV_cache: list of (D,) tensors (oldest -> newest)
          - new_kv: (D,) tensor for the latest step
          - h_state: LSTM state
          - buffer: list collecting recent tokens up to size C

        Behavior:
          1) Append the new token to both KV_cache and the buffer.
          2) If buffer has reached C and the total cache length > W:
             - Run gate on the buffer (shape to (1, C, D)).
             - Compute 'first_idx' = index of the *oldest* chunk just beyond the W window.
             - If gate >= threshold:
                  Replace that chunk with a single aggregated token.
               Else:
                  Drop that chunk entirely.
             - Clear the buffer.

        Returns:
          - KV_cache (possibly modified), updated h_state, buffer
        """
        if buffer is None:
            buffer = []
            
        # Append newest token
        KV_cache.append(new_kv)
        buffer.append(new_kv)
        v_cache.append(new_v)
        buffer_v.append(new_v)
        
        # Only start compress/drop decisions once we have a full chunk
        # and there are tokens beyond the recent window.
        if len(buffer) == self.C and len(KV_cache) > self.W:
            buffer_tensor = torch.stack(buffer, dim=0).unsqueeze(0)  # (1, C, D)
            buffer_tensor_v = torch.stack(buffer_v, dim=0).unsqueeze(0)  # (1, C, D)
            
            agg_chunk = self.aggregate(buffer_tensor.squeeze(0)) # (1, D)
            agg_chunk_v = self.aggregate(buffer_tensor_v.squeeze(0)) # (1, D)
            
            gate_score, h_state = self.rnn_gate(buffer_tensor, h_state)
            

            # Oldest chunk just beyond the recent window.
            first_idx = len(KV_cache) - self.W - self.C
            
            if gate_score >= self.threshold:
                # Keep: compress into one token.
                KV_cache[first_idx] = agg_chunk.squeeze(0)
                v_cache[first_idx] = agg_chunk_v.squeeze(0)
                for _ in range(self.C - 1):
                    del KV_cache[first_idx + 1] # remove the rest
                    del v_cache[first_idx + 1] # remove the rest
                    
            else:
                # Drop the whole chunk.
                for _ in range(self.C):
                    del KV_cache[first_idx]
                    del v_cache[first_idx]
            buffer = []
        return KV_cache, v_cache, h_state, buffer,buffer_v
