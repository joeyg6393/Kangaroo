# Collision Detection Time-Space Trade-offs for Pollard's Kangaroo ECDLP Solver

## 1. Overview

### Why Systematic Time-Space Trade-offs Matter

In Pollard's kangaroo algorithm for solving the discrete logarithm problem, collision detection represents the critical moment when the search concludes—when a tame and wild kangaroo meet at the same point. The method by which collisions are detected, and the resources allocated to this detection, fundamentally determine both the memory requirements and the expected running time of the algorithm.

Time-space trade-offs in collision detection arise because:

1. **Memory limitation**: Storing every point visited by kangaroos would require O(√N) memory, which is prohibitive for large intervals
2. **Detection delay**: Using distinguished points (DPs) means collisions aren't detected immediately—kangaroos continue walking after collision until reaching a DP
3. **Wasted work**: The steps taken after collision but before detection represent "excess iterations" that don't contribute to finding the solution

The distinguished point method, pioneered by van Oorschot and Wiener, provides a elegant trade-off: by only storing points whose x-coordinates have a specific property (e.g., leading zero bits), memory requirements drop dramatically while collision detection remains efficient.

### The Core Trade-off Equation

The fundamental time-space relationship for parallel collision search follows:

```
T² · S = Θ(K² · N)
```

Where:
- **T** = expected running time (group operations)
- **S** = memory (number of distinguished points stored)
- **K** = number of parallel kangaroos
- **N** = size of the search interval

This equation shows that reducing memory by factor c increases time by factor √c—a square-root trade-off that governs all practical implementations.

## 2. Distinguished Point Theory

### How DP Density Affects Performance

A point P is "distinguished" if its x-coordinate satisfies a specific property, typically having dpBit leading zero bits:

```
is_distinguished(P) = (P.x >> (256 - dpBit)) == 0
```

The probability that a random point is distinguished:
```
θ = Pr[point is distinguished] = 1/2^dpBit
```

#### Impact of θ on Algorithm Behavior

**Storage Requirements:**
```
Expected DPs stored ≈ (number of steps) × θ
                    ≈ 2√N × θ (for standard two-kangaroo method)
```

**Detection Delay:**
```
Expected steps after collision until DP = 1/θ = 2^dpBit
```

**Overhead Factor:**
The overhead from using distinguished points follows:
```
overhead = f(w · θ²)
```
Where w is the number of parallel walkers. When w·θ² is small, overhead approaches zero; as it grows, overhead increases cubically.

### The Loop Detection Problem

A critical challenge with distinguished points is cycle detection. If a kangaroo enters a cycle that doesn't contain any distinguished points, it will loop forever without contributing useful work.

**Probability of "useless" cycle:**
- Depends on dpBit and the structure of the iteration function
- For well-designed iteration functions: negligible for dpBit < 30
- Risk increases for very sparse DP conditions (large dpBit)

**Mitigation strategies:**
1. **Timeout detection**: Kill walks that exceed expected length by large factor
2. **Step counters**: Track steps since last DP, restart if threshold exceeded
3. **Hybrid detection**: Combine DP with periodic checkpoints

### Uniform Distribution Properties

Distinguished points should be uniformly distributed across the search space for optimal collision detection:

```
Pr[collision detected | collision occurred] ≈ 1 - e^(-steps_after_collision × θ)
```

After 1/θ steps, detection probability exceeds 63%. After 3/θ steps, it exceeds 95%.

## 3. Time-Space Trade-off Formulas

### Van Oorschot-Wiener Formulas

The classic parallel collision search paper established:

**Expected time for k parallel processors:**
```
T = Θ(√(N/k))
```

**Expected distinguished points stored:**
```
S = Θ(k × √N × θ)
```

**Trade-off curve:**
```
T² × S = Θ(N)  (for optimal θ selection)
```

### Multi-Kangaroo Expected Operations

| Method | Expected Operations | Constant c |
|--------|---------------------|------------|
| Two-kangaroo (Pollard) | 2.0√N | 2.00 |
| Three-kangaroo | 1.818√N | 1.818 |
| Four-kangaroo | 1.714√N | 1.714 |
| Seven-kangaroo | 1.7195√N | 1.7195 |

### Checkpoint Method (Wang-Zhang)

An alternative to pure distinguished points:

**Checkpoint interval:** Store points at regular intervals i
**Expected operations:** Approximately 1.295√N with optimal interval
**Advantage:** No useless cycles possible
**Disadvantage:** More memory than optimal DP method

### Optimal θ Selection

For w parallel walkers with memory budget S:

```
θ_optimal ≈ 2.25 / √w
```

This minimizes total work while maintaining acceptable memory usage.

### Excess Iteration Formulas

After a collision occurs, additional steps before detection:

```
E[excess steps] = 1/θ = 2^dpBit
```

For the entire algorithm:
```
E[total excess] ≈ 2^dpBit × (number of collision opportunities)
```

## 4. Optimal Parameter Selection

### Methodology for Choosing dpBit

Given:
- **N** = search interval size (bits)
- **w** = number of parallel kangaroos
- **M** = available memory (bytes)
- **Entry size** = bytes per DP entry (typically 40-64 bytes)

**Step 1: Calculate maximum DPs that fit in memory**
```
max_DPs = M / entry_size
```

**Step 2: Estimate expected DPs for given dpBit**
```
expected_DPs = w × 2√N / 2^dpBit
```

**Step 3: Choose dpBit such that expected_DPs ≤ max_DPs**
```
dpBit ≥ log₂(w × 2√N / max_DPs)
```

**Step 4: Verify overhead is acceptable**
```
overhead_ratio = w × 2^(-2×dpBit) × N
```
If overhead_ratio > 0.5, consider reducing dpBit or increasing memory.

### Practical Parameter Guidelines

| Interval Size | Recommended dpBit | Expected DPs (w=1000) | Memory |
|---------------|-------------------|----------------------|--------|
| 64-bit | 8-12 | 250K - 16M | 10MB - 640MB |
| 80-bit | 12-16 | 1M - 16M | 40MB - 640MB |
| 96-bit | 16-20 | 4M - 64M | 160MB - 2.5GB |
| 112-bit | 20-24 | 16M - 256M | 640MB - 10GB |
| 128-bit | 24-28 | 64M - 1B | 2.5GB - 40GB |

### Balanced Approach for Constrained Systems

When memory is severely limited:

1. **Start with larger dpBit** (sparser DPs)
2. **Accept higher overhead** (more excess iterations)
3. **Use work file merging** to combine results from multiple runs
4. **Consider checkpoint hybrid** for very long runs

## 5. Checkpoint Variants

### Minimum Value Method

Instead of distinguished points, store points at regular intervals:

**Algorithm:**
```
every I steps:
    store current point and distance
```

**Properties:**
- Uniform memory usage over time
- No cycle-trapping risk
- Deterministic memory requirements
- Slightly higher total storage for same collision probability

### Hybrid Checkpoint-DP

Combine both methods:

```
if is_distinguished(P):
    store P
else if steps_since_last_store > max_interval:
    store P as checkpoint
```

**Benefits:**
- Catches collisions quickly via DPs
- Prevents infinite loops via checkpoints
- Adaptive to actual walk behavior

### Bloom Filter Variant

For extremely memory-constrained scenarios:

**Approach:**
- Use Bloom filter for first-pass collision detection
- Verify candidates with small secondary table
- Trade false positives for memory efficiency

**Parameters:**
- Bloom filter size: m bits
- Number of hash functions: k
- False positive rate: (1 - e^(-kn/m))^k

### Nested DP Levels

Use multiple DP thresholds:

```
Level 1: dpBit = 10 (frequent, stored locally)
Level 2: dpBit = 16 (moderate, stored in RAM)
Level 3: dpBit = 24 (rare, stored in central server)
```

**Advantage:** Hierarchical collision detection with tiered memory

## 6. Reducing Wasted Work

### Sources of Wasted Work

1. **Excess iterations**: Steps after collision but before DP detection
2. **Cycle-trapping**: Walks stuck in DP-free cycles
3. **Duplicate DPs**: Same DP reached by same kangaroo type

### Quantifying Excess Iterations

```
E[excess] ≈ 2^(dpBit-1) per collision
```

For total wasted work:
```
Wasted_fraction = E[excess] / E[total_steps]
                = 2^dpBit / (2√N)
```

**Example:** For 64-bit interval (N = 2^64) with dpBit = 16:
```
Wasted_fraction = 2^16 / 2^33 ≈ 0.002 = 0.2%
```

### Strategies for Minimization

#### 1. Early Termination Heuristics

```c
// Stop walk if collision likely already occurred
if (steps > 3 * expected_steps && no_collision_found) {
    restart_walk_from_new_position();
}
```

#### 2. Buffered DP Transmission

```c
// Batch DP submissions to reduce overhead
if (dp_buffer.size() >= BATCH_SIZE || time_since_last_send > MAX_DELAY) {
    send_batch_to_server(dp_buffer);
    dp_buffer.clear();
}
```

#### 3. Loop Detection

```c
// Detect stuck walks
if (steps_since_last_dp > 10 * expected_dp_interval) {
    // Likely in DP-free cycle
    restart_walk();
}
```

### When Checkpoint Methods Win

Checkpoint methods become superior when:

1. **dpBit is very large** (> 28): Excess iteration cost dominates
2. **Memory is abundant**: Can afford regular checkpoints
3. **Reliability required**: Cannot risk cycle-trapping
4. **Short intervals**: Overhead from checkpoints is relatively small

## 7. Implementation Considerations

### CUDA/GPU Implementation

```cuda
__global__ void kangaroo_step_with_dp(
    point_t* positions,
    uint64_t* distances,
    uint32_t dp_bits,
    dp_entry_t* dp_buffer,
    uint32_t* dp_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform jump
    point_t p = positions[tid];
    uint64_t d = distances[tid];

    uint32_t jump_idx = hash(p.x) & JUMP_MASK;
    p = point_add(p, jump_table[jump_idx]);
    d += jump_distances[jump_idx];

    // Check for distinguished point
    uint64_t top_bits = p.x >> (256 - dp_bits);
    if (top_bits == 0) {
        // Atomic increment to get buffer slot
        uint32_t slot = atomicAdd(dp_count, 1);
        if (slot < DP_BUFFER_SIZE) {
            dp_buffer[slot] = {p.x, d, tid};
        }
    }

    positions[tid] = p;
    distances[tid] = d;
}
```

### Distributed Hash Table Strategies

For multi-machine deployments:

```
┌──────────────────────────────────────────────┐
│           Central DP Server                   │
│  ┌────────────────────────────────────────┐  │
│  │  Partitioned Hash Table               │  │
│  │  Partition 0: hash(x) mod P == 0      │  │
│  │  Partition 1: hash(x) mod P == 1      │  │
│  │  ...                                   │  │
│  │  Partition P-1                         │  │
│  └────────────────────────────────────────┘  │
└───────────────┬──────────────────────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───┴───┐   ┌───┴───┐   ┌───┴───┐
│Client │   │Client │   │Client │
│  0    │   │  1    │   │  N    │
└───────┘   └───────┘   └───────┘
```

### Work File Management

```c
// Work file format for resumable searches
struct WorkFileHeader {
    uint64_t magic;           // File format identifier
    uint32_t version;         // Format version
    uint32_t dp_bits;         // Distinguished point threshold
    uint256_t range_start;    // Search interval start
    uint256_t range_end;      // Search interval end
    uint256_t public_key;     // Target public key
    uint64_t total_ops;       // Operations completed
    uint32_t dp_count;        // Distinguished points stored
    uint32_t kangaroo_count;  // Number of saved kangaroos
};
```

### Configuration Template

```ini
# Kangaroo Configuration
[search]
range_start = 0x1000000000000000
range_end = 0x1FFFFFFFFFFFFFFF
public_key = 0x03...

[parameters]
dp_bits = 16                    # Distinguished point threshold
num_kangaroos = 1024            # Parallel walks
save_interval = 300             # Work file save interval (seconds)

[memory]
hash_table_size = 268435456     # 256M entries
entry_size = 48                 # Bytes per DP entry

[optimization]
max_steps_without_dp = 1000000  # Loop detection threshold
batch_size = 65536              # DP batch transfer size
```

## 8. References

### Primary Papers

1. **van Oorschot, P.C. & Wiener, M.J.** (1999). "Parallel Collision Search with Cryptanalytic Applications." *Journal of Cryptology*, 12(1), 1-28.
   - URL: https://people.scs.carleton.ca/~paulv/papers/JoC97.pdf

2. **Wang, P. & Zhang, F.** (2012). "An Efficient Collision Detection Method for Computing Discrete Logarithms."
   - URL: https://www.hindawi.com/journals/jam/2012/635909/

3. **Fowler, A. & Galbraith, S.D.** (2015). "Kangaroo Methods for Solving the Interval Discrete Logarithm Problem."
   - URL: https://arxiv.org/abs/1501.07019

4. **Hellman, M.E.** (1980). "A Cryptanalytic Time-Memory Trade-Off." *IEEE Transactions on Information Theory*, 26(4), 401-406.

### Implementation References

5. **JeanLucPons/Kangaroo** - GPU-accelerated implementation
   - URL: https://github.com/JeanLucPons/Kangaroo

### Related Work

6. **Time-Memory Trade-offs for Parallel Collision Search**
   - URL: https://eprint.iacr.org/2017/581.pdf

7. **Computing Discrete Logarithms in an Interval**
   - URL: https://eprint.iacr.org/2010/617.pdf

---

## Summary

The key formulas for practical implementation:

| Parameter | Formula | Notes |
|-----------|---------|-------|
| θ_optimal | 2.25 / √w | w = parallel walkers |
| Expected DPs | w × 2√N × θ | Storage requirement |
| Excess iterations | 2^(dpBit-1) | Per collision |
| Memory (bytes) | DPs × entry_size | Typically 40-64 bytes/entry |
| dpBit selection | log₂(w × 2√N / max_DPs) | Based on memory budget |

---

**Document Date**: January 2026
**Status**: Technical reference for time-space trade-off optimization in Pollard's kangaroo algorithm
