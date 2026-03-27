// Copyright 2026 Rubrik, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package table

import (
	"container/list"
	"sync"

	"github.com/golang/leveldb/db"
)

// BlockCacheOption is a functional option for NewLRUBlockCache.
type BlockCacheOption func(*lruBlockCache)

// WithOnClose sets a callback that is invoked in Close() with the
// final cache stats. Use this to log hit/miss rates in production.
// The callback captures the caller's context via closure.
func WithOnClose(fn func(db.BlockCacheStats)) BlockCacheOption {
	return func(c *lruBlockCache) {
		c.onClose = fn
	}
}

// lruEntry holds a cached decompressed block and its offset key.
type lruEntry struct {
	offset uint64
	data   block
}

// lruBlockCache implements db.BlockCache with LRU eviction.
// It uses a hash map for O(1) lookups and a doubly-linked list
// for O(1) LRU ordering. All operations are goroutine-safe.
type lruBlockCache struct {
	mu        sync.Mutex
	items     map[uint64]*list.Element // offset → list element
	order     *list.List               // front = most recent, back = least recent
	capacity  int
	hits      int64
	misses    int64
	evictions int64
	onClose   func(db.BlockCacheStats) // optional callback on Close
}

// NewLRUBlockCache creates a block cache with LRU eviction.
// capacity is the maximum number of decompressed blocks to hold.
// Each block is typically 512 KiB, so capacity=16 ≈ 8 MB.
// Optional functional options can be passed to configure behavior
// (e.g., WithOnClose for logging stats on close).
func NewLRUBlockCache(capacity int, opts ...BlockCacheOption) db.BlockCache {
	c := &lruBlockCache{
		items:    make(map[uint64]*list.Element, capacity),
		order:    list.New(),
		capacity: capacity,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Get returns the cached block for the given offset, or nil if
// not cached. On a hit, the entry is moved to the front of the
// LRU list (most recently used).
func (c *lruBlockCache) Get(offset uint64) []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	if elem, ok := c.items[offset]; ok {
		c.hits++
		c.order.MoveToFront(elem)
		return elem.Value.(*lruEntry).data
	}
	c.misses++
	return nil
}

// Put stores a decompressed block in the cache. If the cache is
// full, the least recently used entry is evicted first.
func (c *lruBlockCache) Put(offset uint64, data []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Already cached (e.g., another goroutine cached it concurrently).
	if _, ok := c.items[offset]; ok {
		return
	}

	// Evict least recently used if at capacity.
	if c.order.Len() >= c.capacity {
		evicted := c.order.Back()
		evictedEntry := evicted.Value.(*lruEntry)
		delete(c.items, evictedEntry.offset)
		c.order.Remove(evicted)
		c.evictions++
	}

	// Insert at front (most recently used).
	entry := &lruEntry{offset: offset, data: data}
	elem := c.order.PushFront(entry)
	c.items[offset] = elem
}

// Stats returns current cache hit/miss/eviction counters.
func (c *lruBlockCache) Stats() db.BlockCacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return db.BlockCacheStats{
		Hits:      c.hits,
		Misses:    c.misses,
		Evictions: c.evictions,
		Entries:   c.order.Len(),
	}
}

// Close releases all cached blocks and resources. If an OnClose
// callback was configured, it is called with the final stats
// before releasing memory.
func (c *lruBlockCache) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.onClose != nil {
		c.onClose(db.BlockCacheStats{
			Hits:      c.hits,
			Misses:    c.misses,
			Evictions: c.evictions,
			Entries:   c.order.Len(),
		})
	}
	c.items = nil
	c.order = nil
}
