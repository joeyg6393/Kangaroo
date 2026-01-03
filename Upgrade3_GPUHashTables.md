# Improved Collision/Distinguished-Point Handling via GPU/CPU Hash Tables

## Overview

Distinguished Points (DPs) are a critical optimization in Pollard's kangaroo algorithm for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP). Instead of storing all points in random walks, only points with specific properties (those with x-coordinates starting with a predetermined number of zero bits) are stored. When two kangaroos collide, they follow the same path thereafter due to the deterministic jump function.

Hash tables are the essential data structure for managing distinguished points, detecting collisions between tame and wild kangaroo herds, and performing rapid lookups across parallel computation streams. The efficiency of these hash tables directly impacts:

- **Memory consumption**: Large DP tables can exhaust GPU memory quickly
- **Collision detection latency**: Query performance determines the critical path in kangaroo solvers
- **Scalability**: The ability to handle millions to billions of distinguished points across multi-GPU systems
- **Load factors**: Operating at high occupancy (0.90-0.99) without performance degradation

Modern GPU hash table designs—particularly Bucketed Cuckoo Hash Tables (BCHT), compact iceberg hashing, and hybrid CPU-GPU approaches—enable kangaroo solvers to operate at previously unattainable scales while maintaining low memory footprint and high throughput.

## Why Better Hash Tables Matter for Distinguished Points

### The Distinguished Point Problem

In parallel kangaroo solvers, the distinguished point ratio θ determines how frequently points are stored:

```
Storage frequency = θ
Average jumps between DPs = 1/θ
```

If θ is too large (few distinguished points):
- Few collisions detected, prolonging the search
- Reduced collision detection sensitivity

If θ is too small (many distinguished points):
- Rapid hash table saturation
- Memory exhaustion on GPU
- Degraded hash table performance under high load

### Hash Table as the Bottleneck

The central DP hash table becomes a bottleneck in several scenarios:

1. **GPU-to-CPU Transfer**: Thousands of distinguished points must be transferred from GPU to central CPU table per second
2. **Collision Detection**: Every DP lookup must be fast; slow lookups delay kangaroo progress
3. **Memory Bandwidth**: With billions of distinguished points, memory efficiency is paramount
4. **Load Factor**: High occupancy rates (approaching 1.0) are necessary to fit large search spaces in limited GPU memory

### Advanced Hash Table Benefits

State-of-the-art hash table designs provide:

- **Higher load factors**: 0.99 instead of 0.50-0.70, reducing memory overhead
- **Reduced probes**: Fewer memory accesses per lookup = lower latency
- **GPU optimization**: Cache-line aligned buckets, SIMD-friendly operations
- **Hybrid operation**: Seamless GPU-CPU coordination for distributed search
- **Dynamic resizing**: Adaptive capacity without full rebuilds

## Key Hash Table Designs

### 1. Bucketed Cuckoo Hash Tables (BCHT)

#### Concept

BCHT extends classical cuckoo hashing by dividing the hash table into buckets, each capable of storing multiple keys. Instead of single-slot collision resolution, buckets hold several elements (typically 4-16), reducing both the number of hash functions needed and probe count.

#### Key Characteristics

- **Hash functions**: Typically 3-5 independent hash functions
- **Bucket size**: 16-element buckets offer optimal GPU performance
- **Load factor**: Achieves 0.99 with only 3 hash functions
- **Probe count**: Average 1.43 probes per insertion at 0.99 load factor

#### Performance at Various Load Factors

| Load Factor | Avg Probes (Insert) | Avg Probes (Query+) | Avg Probes (Query-) |
|---|---|---|---|
| 0.50 | 1.03 | 1.02 | 1.05 |
| 0.70 | 1.10 | 1.07 | 1.25 |
| 0.90 | 1.30 | 1.20 | 2.20 |
| 0.99 | 1.43 | 1.39 | 2.80 |

#### GPU Optimization for BCHT

BCHT is particularly well-suited to GPU architectures because:

1. **Cache-line alignment**: 16-element buckets typically align with GPU cache lines (64-128 bytes)
2. **Coalesced memory access**: All bucket contents can be fetched in a single transaction
3. **Reduced branch divergence**: Multiple elements per bucket amortize control flow overhead
4. **Scalability**: Thread-parallel bucket operations minimize synchronization

### 2. Iceberg Hashing

#### Concept

Iceberg hashing uses a multi-level architecture with routing tables that direct lookups to candidate buckets. It provides:

- **Stable operation**: Performance guarantees without dynamic resizing
- **Cache efficiency**: (1 + o(1)) cache guarantees reduce TLB misses
- **Compact form**: Can store more entries per cache line

#### Performance Characteristics

- **Load factor**: Up to 0.91 with bucket size 32
- **Space efficiency**: Operates at >85% full without performance loss
- **Lookup cost**: Nearly constant, predictable latency
- **Dynamic operation**: Supports insertions without full rebuild

#### Iceberg vs. Cuckoo for Distinguished Points

| Criterion | Bucketed Cuckoo | Iceberg |
|---|---|---|
| Max load factor | 0.99 | 0.91 |
| Lookup stability | Variable | Predictable |
| Memory hierarchy | Cache-line aligned | TLB efficient |
| Dynamic operations | Requires rehashing | Continuous |
| Compact variant | 10-20% better | Significant improvement |

### 3. Compact Hash Tables

#### What Makes Tables "Compact"

Compact hashing techniques store more entries per cache line by:

1. Using variable-length encodings for keys/values
2. Eliminating padding in bucket structures
3. Optimizing for common case (small keys/values)
4. Reducing wasted space in partially-filled buckets

#### Performance Improvements

Recent research (Hegeman et al., 2024) demonstrates:

- **Cuckoo hashing**: 10-20% throughput increase with compactness
- **Iceberg hashing**: Significant improvements, achieving parity with compact cuckoo
- **Memory efficiency**: Reduced memory bandwidth requirements

## GPU-Side Buffering Strategy

### Architecture Overview

GPU-side buffering uses a two-tier approach:

1. **Local DP Buffer** (GPU shared memory / L1 cache):
   - Compact hash table holding recently detected distinguished points
   - Very high bandwidth, very limited space (96KB - 256KB per SM)
   - Holds 1K-10K entries depending on key/value size

2. **Global DP Table** (GPU VRAM):
   - Bucketed cuckoo or compact iceberg table
   - High bandwidth but higher latency than shared memory
   - Holds millions to billions of distinguished points
   - Load factor optimized (0.90-0.99)

### Buffering Workflow

```
Kangaroo Thread                GPU Hash Table              CPU/Host
    |                              |                          |
    +------- DP detected -------->  |                          |
    |      (x-coordinate match)     |                          |
    |                          Local Buffer?                   |
    |                              |                          |
    |                    (if hit) Return metadata             |
    |                    (if miss) Insert locally              |
    |                              |                          |
    |                     Buffer overflow?                     |
    |                              |                          |
    |                  (Periodic flush to global)              |
    |                              +---- Batch transfer ------>|
    |                              |                          |
    |                         Global lookup?                   |
    |                              |                          |
    |                    (hit) Collision detected              |
    |<---------- Return collision info ------                  |
    |                              |                          |
    +---- Continue walk ---------> |                          |
```

### Double-Buffering for Batch Transfer

To maximize PCIe bandwidth utilization while maintaining GPU compute throughput:

```c
// Pseudo-code for double-buffered DP transfer
BufferA = GPU_DP_buffer;
BufferB = GPU_DP_buffer;

while (solver_running) {
    // GPU kernels fill one buffer
    LaunchKangarooKernel(outputBuffer: BufferA);

    // Asynchronously transfer other buffer to CPU
    TransferBuffer(BufferB, GPU_TO_CPU, stream_1);

    // CPU processes received DPs while GPU computes
    ProcessAndInsertDPs(BufferB, cpu_hash_table);

    // Swap buffers
    swap(BufferA, BufferB);

    // Synchronize before next iteration
    WaitForGPUKernel();
    WaitForCPUTransfer();
}
```

### Key Performance Metrics

- **Local buffer hit rate**: 5-15% of lookups hit local buffer (10-100x latency improvement)
- **Global buffer efficiency**: Batched transfers achieve 90%+ of peak PCIe bandwidth
- **Transfer throughput**: Modern PCIe Gen4: ~15 GB/s; Gen5: ~32 GB/s
- **Batch size**: 64K-256K distinguished points per transfer optimizes transfer-to-compute ratio

## CPU-Side Tables

### Purpose and Design

The CPU-side hash table serves as the central collision repository and accepts batched distinguished point uploads from all GPU devices in a multi-GPU system. Design priorities:

1. **High load factor**: 0.90-0.99 to minimize memory consumption
2. **Batch insertion efficiency**: Optimized for bulk operations, not single-element insertion
3. **Query throughput**: Support rapid collision detection queries from GPU pull-backs
4. **Dynamic growth**: Resizing capability without system pause

### Heterogeneous CPU-GPU Hash Table Architecture

Recent research demonstrates that hybrid CPU-GPU hash tables optimize for non-uniform access patterns:

```
Access Pattern Distribution:
┌──────────────────────────────────┐
│ Hot keys (1-10% of all keys)    │ ──-> GPU Table
│ Accessed 90%+ of the time        │
├──────────────────────────────────┤
│ Warm keys (10-30%)               │ ──-> CPU Cache
│ Accessed occasionally            │
├──────────────────────────────────┤
│ Cold keys (60-90%)               │ ──-> CPU Main Memory
│ Rarely accessed after insertion  │
└──────────────────────────────────┘
```

### Batch Processing Strategy

CPU-side tables employ batched operations for efficiency:

```
Batch Operation Performance (64K-256K DPs per batch):
┌──────────────────┬─────────────────┬──────────────────┐
│ Operation        │ Throughput      │ Latency (bulk)   │
├──────────────────┼─────────────────┼──────────────────┤
│ Insert (batch)   │ 1-2 M ops/sec   │ 100-200 ms       │
│ Query (batch)    │ 5-10 M ops/sec  │ 20-50 ms         │
│ Mixed ops        │ 2-5 M ops/sec   │ 50-100 ms        │
└──────────────────┴─────────────────┴──────────────────┘
```

## Performance Characteristics

### Throughput Benchmarks

#### GPU Hash Table Operations (Modern GPUs: RTX 4090, H100)

| Operation | BCHT (0.99 LF) | Iceberg (0.91 LF) | WarpCore | Hive |
|---|---|---|---|---|
| Insert | 1.2-1.6 B/sec | 0.8-1.0 B/sec | 1.6 B/sec | 2.0-3.5 B/sec |
| Lookup | 2.0-3.0 B/sec | 1.5-2.2 B/sec | 4.3 B/sec | 3.5-4.0 B/sec |
| Mixed | 1.0-1.5 B/sec | 0.9-1.3 B/sec | 1.2 B/sec | 1.5-2.5 B/sec |

### Memory Efficiency

#### Distinguished Point Storage Requirements

For a typical 130-bit search range with θ = 2^-24 (distinguished point ratio):

```
At different load factors:
┌──────────┬─────────────────┬────────────┐
│ Load LF  │ Table size      │ GPU memory │
├──────────┼─────────────────┼────────────┤
│ 0.50     │ 2^107 entries   │ 512 GB     │
│ 0.70     │ 1.43 * 2^106    │ 360 GB     │
│ 0.90     │ 1.11 * 2^106    │ 280 GB     │
│ 0.99     │ 1.01 * 2^106    │ 256 GB     │
└──────────┴─────────────────┴────────────┘

Memory savings from high load factor:
0.50 LF → 0.99 LF: 2x reduction in storage
```

#### Compact Hash Table Advantage

Compact tables reduce memory overhead by 15-30% compared to standard bucketed tables:

```
Standard Bucket (16 slots):
┌─────────────────────────────────┐
│ Bucket header         │ 16 bytes │
│ Key 1 + Value 1       │ 40 bytes │
│ ...                   │ ...      │
│ Padding               │ 16 bytes │
├─────────────────────────────────┤
│ Total per bucket      │ 512 bytes│
│ Efficiency            │ ~85%     │
└─────────────────────────────────┘

Compact Bucket:
┌─────────────────────────────────┐
│ Encoded keys (variable length)  │
│ Inline values (compact)         │
│ Offset table (16 bytes)         │
├─────────────────────────────────┤
│ Total per cache line  │ 64 bytes │
│ Entries per CL        │ 12-16    │
│ Efficiency            │ ~95%     │
└─────────────────────────────────┘

Result: 15-30% memory savings for same capacity
```

## Implementation Considerations

### Integration Points with Kangaroo Solver

#### Distinguished Point Detection

```cuda
__global__ void kangaroo_step_with_dp_detection(
    point_t* positions,           // Current kangaroo positions
    uint64_t* distances,          // Walk distances
    dp_buffer_t* local_dp_buffer, // Shared memory DP cache
    gpu_ht_t* global_ht,          // Global DP hash table
    collision_t* collisions       // Output collisions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform kangaroo jump
    point_t new_pos = jump(positions[tid], distances[tid]);
    distances[tid]++;

    // Check if distinguished point
    uint64_t x_coord = new_pos.x;
    if (is_distinguished(x_coord, DP_BITS)) {
        // Try local buffer first
        dp_metadata_t* local_entry = local_dp_buffer.query(x_coord);

        if (local_entry != NULL) {
            // Collision detected in local buffer
            record_collision(collisions, tid, *local_entry);
        } else {
            // Check global table
            dp_metadata_t* global_entry = global_ht.query(x_coord);

            if (global_entry != NULL) {
                record_collision(collisions, tid, *global_entry);
            } else {
                // New distinguished point
                dp_metadata_t meta = {x_coord, tid, distances[tid]};
                global_ht.insert(x_coord, meta);
            }
        }
    }

    positions[tid] = new_pos;
}
```

### Hash Function Selection

For distinguished point tables, use high-quality hash functions:

**Recommended**: xxHash, MurmurHash3, or cryptographic hash for security
**Avoid**: Simple modular arithmetic, linear probing

```cuda
// Example: Three independent hash functions for BCHT
__device__ uint64_t hash_func_1(uint64_t key) {
    return xxhash64(key, SEED_1);
}

__device__ uint64_t hash_func_2(uint64_t key) {
    return xxhash64(key, SEED_2);
}

__device__ uint64_t hash_func_3(uint64_t key) {
    return xxhash64(key, SEED_3);
}

// Compute bucket positions
uint64_t h1 = hash_func_1(key) % table_size;
uint64_t h2 = hash_func_2(key) % table_size;
uint64_t h3 = hash_func_3(key) % table_size;
```

### Synchronization and Consistency

#### Thread Safety Guarantees

GPU hash tables (BGHT, WarpCore) are:
- **Lock-free**: No mutex/semaphore overhead
- **Atomic operations**: CAS-based insertion/update
- **Memory consistent**: Proper GPU memory barriers

## Integration Architecture

### Complete System Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    Multi-GPU Kangaroo Solver                   │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU 0                    GPU 1                    GPU N        │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────┐│
│  │ Kangaroo Kernels │    │ Kangaroo Kernels │    │   ...      ││
│  └────────┬─────────┘    └────────┬─────────┘    └──────┬─────┘│
│           │                       │                      │       │
│           ▼                       ▼                      ▼       │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────┐│
│  │Local DP Buffer   │    │Local DP Buffer   │    │   ...      ││
│  │(Shared Memory)   │    │(Shared Memory)   │    │            ││
│  └────────┬─────────┘    └────────┬─────────┘    └──────┬─────┘│
│           │                       │                      │       │
│           └────────────┬──────────┴──────────┬───────────┘       │
│                        ▼                     ▼                   │
│           ┌────────────────────────────────────────┐            │
│           │  Global GPU DP Hash Table             │            │
│           │  Bucketed Cuckoo (0.95 LF)           │            │
│           │  BCHT: 1.6B inserts, 3B queries/sec  │            │
│           └────────────┬─────────────────────────┘            │
│                        │                                       │
│         ┌──────────────┴──────────────┐                        │
│         │ Double-buffered batch      │                        │
│         │ transfer via PCIe/NVLink   │                        │
│         ▼                            ▼                        │
└─────────────────────────┼────────────────────────────────────┘
                          │
                   PCIe / NVLink
                   15-32 GB/sec
                          │
┌─────────────────────────┴────────────────────────────────────┐
│                       CPU Host                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CPU DP Hash Table (Heterogeneous)                   │   │
│  │  Load factor: 0.90-0.99                              │   │
│  │  Throughput: 1-2 B inserts, 5-10 B queries/sec      │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

## References

### Primary Research Papers

1. **Awad, M.A., Ashkiani, S., et al.** (2021). "Analyzing and Implementing GPU Hash Tables."
   Available: https://arxiv.org/pdf/2108.07232

2. **Hegeman, S., Wöltgens, D., et al.** (2024). "Compact Parallel Hash Tables on the GPU."
   Available: https://arxiv.org/abs/2406.09255
   GitHub: https://github.com/system-verification-lab/compact-parallel-hash-tables

3. **Demeyer, H., Tsvyatkova, A., & Genet, B.** (2024). "Hashinator: A Portable Hybrid Hashmap."
   Available: https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1407365/full

### Implementation References

- **BGHT (Better GPU Hash Tables)**: https://github.com/owensgroup/BGHT
- **WarpCore**: https://github.com/sleeepyjack/warpcore
- **cuCollections**: https://github.com/nvidia-labs/cccl

### Kangaroo Algorithm References

- **JeanLucPons/Kangaroo**: https://github.com/JeanLucPons/Kangaroo

---

*Document generated for research and implementation reference. January 2026.*
