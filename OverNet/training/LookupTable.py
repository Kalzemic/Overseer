from collections import deque
import torch


class LookupTable:
    def __init__(self, vocab_size=2**13):
        self.vocab_size = vocab_size
        self.ip_map = {}
        self.page_map = {}
        self._ip_ring = deque()
        self._page_ring = deque()
        self._cache_0 = None
        self._cache_1 = None

    def reset_vocab(self):
        self.ip_map.clear()
        self.page_map.clear()
        self._ip_ring.clear()
        self._page_ring.clear()
        self._cache_0 = None
        self._cache_1 = None

    def _update_cache(self, feat_idx, mapping, device):
        keys = torch.tensor(list(mapping.keys()), dtype=torch.long)
        vals = torch.tensor(list(mapping.values()), dtype=torch.long)
        order = keys.argsort()
        cache = (keys[order].to(device), vals[order].to(device))
        if feat_idx == 0:
            self._cache_0 = cache
        else:
            self._cache_1 = cache
        return cache

    def _update_mapping(self, mapping, ring, col_cpu):
        changed = False
        for v in torch.unique(col_cpu).tolist():
            if v not in mapping:
                if len(mapping) >= self.vocab_size:
                    oldest = ring.popleft()
                    evicted_idx = mapping.pop(oldest)
                    mapping[v] = evicted_idx
                else:
                    mapping[v] = len(mapping)
                ring.append(v)
                changed = True
        return changed

    def lookup(self, data: torch.Tensor) -> torch.Tensor:
        result = torch.empty(data.shape, dtype=torch.long, device=data.device)
        mappings = [self.ip_map, self.page_map]
        rings    = [self._ip_ring, self._page_ring]

        for feat_idx, (mapping, ring) in enumerate(zip(mappings, rings)):
            col   = data[:, :, feat_idx].reshape(-1).to(torch.long).contiguous()
            cache = self._cache_0 if feat_idx == 0 else self._cache_1

            # fast path — try GPU-only lookup if cache exists and is non-empty
            if cache is not None and len(cache[0]) > 0:
                keys_sorted, vals_sorted = cache
                indices = torch.searchsorted(keys_sorted, col)
                clamped = indices.clamp(max=len(keys_sorted) - 1)
                if (keys_sorted[clamped] == col).all():
                    result[:, :, feat_idx] = vals_sorted[indices].reshape(data.shape[0], data.shape[1])
                    continue

            # slow path — new values present, update mapping on CPU
            changed = self._update_mapping(mapping, ring, col.cpu())
            if changed or cache is None:
                cache = self._update_cache(feat_idx, mapping, data.device)

            keys_sorted, vals_sorted = cache
            indices = torch.searchsorted(keys_sorted, col)
            result[:, :, feat_idx] = vals_sorted[indices].reshape(data.shape[0], data.shape[1])

        return result

# from collections import deque
# import torch


# class LookupTable:
#     def __init__(self, vocab_size=2**13):
#         self.vocab_size = vocab_size
#         self.ip_map = {}    # value -> rank
#         self.page_map = {}  # value -> rank
#         self.ip_counts = {}    # value -> frequency count
#         self.page_counts = {}  # value -> frequency count
#         self._cache_0 = None
#         self._cache_1 = None

#     def reset_vocab(self):
#         self.ip_map.clear()
#         self.page_map.clear()
#         self.ip_counts.clear()
#         self.page_counts.clear()
#         self._cache_0 = None
#         self._cache_1 = None

#     def _update_cache(self, feat_idx, mapping, device):
#         keys = torch.tensor(list(mapping.keys()), dtype=torch.long)
#         vals = torch.tensor(list(mapping.values()), dtype=torch.long)
#         order = keys.argsort()
#         cache = (keys[order].to(device), vals[order].to(device))
#         if feat_idx == 0:
#             self._cache_0 = cache
#         else:
#             self._cache_1 = cache
#         return cache

#     def _update_mapping(self, mapping, counts, col_cpu):
#         changed = False
#         vals, freqs = torch.unique(col_cpu, return_counts=True)
#         vals = vals.tolist()
#         freqs = freqs.tolist()

#         for v, f in zip(vals, freqs):
#             counts[v] = counts.get(v, 0) + f

#         for v, f in zip(vals, freqs):
#             if v not in mapping:
#                 if len(mapping) >= self.vocab_size:
#                     # evict the least-frequently-used entry currently in the table
#                     lfu_value = min(mapping.keys(), key=lambda k: counts.get(k, 0))
#                     if counts.get(v, 0) <= counts.get(lfu_value, 0):
#                         # new value isn't even more frequent than the current LFU entry yet
#                         # skip giving it a slot this round; it'll be picked up once its
#                         # count overtakes something already in the table
#                         continue
#                     evicted_idx = mapping.pop(lfu_value)
#                     mapping[v] = evicted_idx
#                 else:
#                     mapping[v] = len(mapping)
#                 changed = True
#         return changed

#     def lookup(self, data: torch.Tensor) -> torch.Tensor:
#         result = torch.empty(data.shape, dtype=torch.long, device=data.device)
#         mappings = [self.ip_map, self.page_map]
#         counts_list = [self.ip_counts, self.page_counts]

#         for feat_idx, (mapping, counts) in enumerate(zip(mappings, counts_list)):
#             col   = data[:, :, feat_idx].reshape(-1).to(torch.long).contiguous()
#             cache = self._cache_0 if feat_idx == 0 else self._cache_1

#             # always update counts so frequency tracking stays accurate,
#             # even for values already mapped
#             col_cpu = col.cpu()
#             changed = self._update_mapping(mapping, counts, col_cpu)

#             if cache is None or len(cache[0]) == 0:
#                 cache = self._update_cache(feat_idx, mapping, data.device)

#             keys_sorted, vals_sorted = cache
#             indices = torch.searchsorted(keys_sorted, col)
#             clamped = indices.clamp(max=len(keys_sorted) - 1)

#             if not (keys_sorted[clamped] == col).all() or changed:
#                 cache = self._update_cache(feat_idx, mapping, data.device)
#                 keys_sorted, vals_sorted = cache
#                 indices = torch.searchsorted(keys_sorted, col)

#             result[:, :, feat_idx] = vals_sorted[indices].reshape(data.shape[0], data.shape[1])

#         return result