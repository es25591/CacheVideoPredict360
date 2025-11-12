import cachetools
import requests


class LruCache:
    def __init__(
        self, 
        max_size=1000, 
        cache_dir='./cache', 
        endpoit='http://10.0.1.51'
    ):
        self.cur_size = 0
        self.max_size = max_size

        self.cache = cachetools.LRUCache(maxsize=self.max_size)
        self.session = requests.Session()

        self.endpoint = endpoit

    def get(self, key, video):
        try:
            return self.cache[key], True, 200
        except KeyError:
            response = self.session.get(f"{self.endpoint}/{key}")
            self.put(key, response.content)

            return response.content, False, 200

    def put(self, key, value):
        self.cache[key] = value
        self.cur_size += len(value)

        if self.cur_size > self.max_size:
            self._spill_to_disk()

    def _spill_to_disk(self):
        while self.cur_size > self.max_size:
            key, value = self.cache.popitem()                
            self.cur_size -= len(value)

    def clear(self):
        self.cache.clear()
        self.cur_size = 0

    def filecache_capacity(self, capacity: int) -> None:
        self.max_size = capacity

    def filecache_is_empty(self):
        return len(self.cache) == 0
    
    def filecache_pop(self) -> str:
        key, value = self.cache.popitem()
        self.cur_size -= len(value)
        return "Removed:" + key
    