import torch
import torch.nn as nn
class SequentialRNNGate(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, chunk, h_prev):
        output, h_next = self.rnn(chunk, h_prev)
        last_out = output[:, -1, :]
        gate_score = torch.sigmoid(self.fc(last_out)).squeeze()
        return gate_score, h_next

def aggregate_mean(chunk):
    if chunk.dim() == 2:
        return chunk.mean(dim=0, keepdim=True)
    else:
        return chunk.mean(dim=1, keepdim=True)

class KVCacheManager:
    def __init__(self, window_size, chunk_size, aggregate_fn, rnn_gate, threshold):
        self.W = window_size
        self.C = chunk_size
        self.aggregate = aggregate_fn
        self.rnn_gate = rnn_gate
        self.threshold = threshold

    def compress_prefill_kv(self, seq_KV, h_state):
        N, D = seq_KV.shape
        retained = []
        for i in range(0, N - self.W, self.C):
            chunk = seq_KV[i : i + self.C]
            agg_chunk = self.aggregate(chunk)
            gate_score, h_state = self.rnn_gate(chunk.unsqueeze(0), h_state)
            if gate_score >= self.threshold:
                retained.append(agg_chunk.squeeze(0))
        for i in range(N - self.W, N):
            retained.append(seq_KV[i])
        return torch.stack(retained, dim=0), h_state

    def online_update(self, KV_cache, new_kv, h_state, buffer=None):
        if buffer is None:
            buffer = []
        KV_cache.append(new_kv)
        buffer.append(new_kv)
        if len(buffer) == self.C and len(KV_cache) > self.W:
            buffer_tensor = torch.stack(buffer, dim=0).unsqueeze(0)  # (1, C, d_k)
            agg_chunk = self.aggregate(buffer_tensor.squeeze(0))
            gate_score, h_state = self.rnn_gate(buffer_tensor, h_state)
            first_idx = len(KV_cache) - self.W - self.C
            if gate_score >= self.threshold:
                KV_cache[first_idx] = agg_chunk.squeeze(0)
                for _ in range(self.C - 1):
                    del KV_cache[first_idx + 1]
            else:
                for _ in range(self.C):
                    del KV_cache[first_idx]
            buffer = []
        return KV_cache, h_state, buffer
