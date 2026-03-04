from collections import defaultdict, OrderedDict


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.min_freq = 0
        self.key_table = {}
        self.freq_table = defaultdict(OrderedDict)

    class Node:
        def __init__(self, key):
            self.key = key
            self.freq = 1

    def _update_freq(self, node):
        del self.freq_table[node.freq][node.key]

        if not self.freq_table[node.freq]:
            del self.freq_table[node.freq]

            if node.freq == self.min_freq:
                self.min_freq += 1

        node.freq += 1
        self.freq_table[node.freq][node.key] = node

    def access(self, key: str):
        if key in self.key_table:
            node = self.key_table[key]
            self._update_freq(node)

        else:
            if self.size >= self.capacity:
                # Evict LFU key (oldest in min_freq)
                k, node_to_remove = self.freq_table[self.min_freq].popitem(last=False)
                del self.key_table[k]
                self.size -= 1

            new_node = self.Node(key)
            self.key_table[key] = new_node
            self.freq_table[1][key] = new_node
            self.min_freq = 1
            self.size += 1

    def contains(self, key: str) -> bool:
        return key in self.key_table

    def get_keys(self):
        return list(self.key_table.keys())
