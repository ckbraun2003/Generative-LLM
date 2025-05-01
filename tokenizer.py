class BPE:

    def __init__(self, data: str, vocab_size=500):
        self.training_tokens = data.encode("utf-8")
        self.training_tokens = list(map(int, self.training_tokens))
        self.vocab_size = vocab_size

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, verbose=True):
        num_merges = self.vocab_size - 256
        self.merges = {}

        for i in range(num_merges):
            stats = self.get_stats(self.training_tokens)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            if verbose:
                print(f"Merging {pair} into new token {idx}")
            self.training_tokens = self.merge(self.training_tokens, pair, idx)
            self.merges[pair] = idx

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text