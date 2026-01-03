# Collision Detection Time-Space Trade-offs: Pollard's Kangaroo ECDLP Solver

Comprehensive Analysis of Distinguished Point Methods and Optimization Strategies

---

## Table of Contents

1. [Overview](#overview)
2. [Distinguished Point Theory](#distinguished-point-theory)
3. [Time-Space Trade-off Formulas](#time-space-trade-off-formulas)
4. [Optimal Parameter Selection](#optimal-parameter-selection)
5. [Checkpoint Variants](#checkpoint-variants)
6. [Reducing Wasted Work](#reducing-wasted-work)
7. [Implementation Considerations](#implementation-considerations)
8. [References](#references)

---

## Overview

### Why Systematic Time-Space Trade-offs Matter

Pollard's kangaroo algorithm solves the Elliptic Curve Discrete Logarithm Problem (ECDLP) with expected complexity of **O(√N)** group operations, where N is the search interval size. However, naive implementation requires storing all intermediate points, making it memory-prohibitive for practical applications.

**The Core Challenge**: When computing discrete logarithms in cyclic groups of large orders using Pollard's rho/kangaroo method, collision detection is invariably a high time and space consumer.

**The Solution**: Systematic time-space trade-offs allow us to:

- Minimize memory footprint while maintaining algorithmic efficiency
- Detect collisions without storing complete paths
- Balance communication overhead (in parallel settings) against computational cost
- Scale from single-processor to distributed GPU-accelerated environments

### Historical Context

The evolution of collision detection methods:

1. **1978**: Pollard introduces the basic kangaroo method with ~3.3√N expected operations
2. **1982**: Rivest introduces distinguished points, drastically reducing memory lookups
3. **1999**: van Oorschot and Wiener provide parallel collision search framework with ~2√N operations
4. **2010**: Galbraith et al. reduce to 1.818√N (three-kangaroo) and 1.714√N (four-kangaroo)
5. **2012**: Wang and Zhang introduce optimal minimum-value checkpoint methods
6. **2015**: Fowler and Galbraith achieve 1.715√N with seven-kangaroo variants

---

## Distinguished Point Theory

### The Distinguished Point Method

Instead of storing every point along the random walk, the distinguished point (DP) method stores only points satisfying a specific property: **having a designated number of leading zero bits in the x-coordinate**.

#### How It Works

1. **Definition**: A point P with x-coordinate x is "distinguished" if the first d bits of x equal zero (or equivalently, if x < 2^(b-d) for a b-bit field)

2. **Storage**: When a kangaroo generates a point that is distinguished, its coordinates are recorded in a hash table

3. **Collision Detection**: When two kangaroos collide at some point C, they follow identical subsequent paths (since jump distances depend only on x-coordinate values). The collision is detected when both reach a distinguished point

4. **Advantage**: Memory complexity drops from O(√N) to O(√N/θ) where θ is the DP density

### DP Density and Parameters

#### Key Definitions

- **θ (theta)**: The proportion of distinguished points in the total point space
  - Mathematically: θ = 1/2^d, where d = dpBit (number of leading zero bits)

- **dpBit**: The number of leading bits that must be zero
  - Larger dpBit → Fewer stored points → Lower memory but more operations to find DP
  - Smaller dpBit → More stored points → Higher memory but faster collision detection

#### DP Density Impact on Performance

**Memory Usage**: W = √N·θ^(-1/2) (approximate expected DP storage)

**Overhead Factor**: O(nbKangaroo · 2^dpBit) steps until collision detected

**Expected Storage**: For random mappings, ~√(2d) steps between collision and detection

#### The Uniform Distribution Property

A critical insight: Using minimum values stored at interval N creates **equal-interval distribution** of checkpoints, whereas distinguished points may not be uniformly distributed in pseudo-random walks. This distinction drives the performance difference between DP and checkpoint methods.

### Loop Detection Problem

**Fundamental Issue**: A random walk can fall into a cycle containing no distinguished points. When this occurs:
- The processor cannot find new DP anymore on the path
- Without detection, the processor ceases contributing to collision search
- The solution: either restart with new initial point or use checkpoint methods (see Section 5)

---

## Time-Space Trade-off Formulas

### Van Oorschot-Wiener Parallel Collision Search

The foundational framework for time-space optimization:

**Time-Space Trade-off Curve**:
```
T² · S = Θ(K² · N)
```

Where:
- T = time (number of group operations)
- S = space (memory in elements)
- K = number of collision pairs to find
- N = size of the search space

**For single collision (K=1)**:
```
T² · S ≈ N²
```

### Distinguished Point Method Complexity

**Basic Kangaroo (2 kangaroos)**:
```
Expected Operations = 2√N
Expected DP Storage = 2√(N/θ) = 2√N · 2^(d/2)
```

Where d = number of leading zero bits

**Multi-kangaroo Variants**:
- Three-kangaroo: (1.818 + o(1))√N operations, (1.818...)√(N/θ) storage
- Four-kangaroo: (1.714 + o(1))√N operations, (1.714...)√(N/θ) storage
- Seven-kangaroo: (1.7195 + o(1))√N operations

### Minimum Value Checkpoint Method

**Wang-Zhang Algorithm (2012)**: By storing minimum values at regular intervals N:

**Expected Operations** (compared to DP method):
```
Checkpoint: 1.295√|G|
Distinguished Points: 1.309√|G|
Reduction: ~1% improvement for equivalent storage
```

**Storage Requirement**:
```
S_checkpoint = (Number of kangaroos) × (Number of steps to find collision) / N
= O(√N / N) with proper N selection
```

**Key Property**: Checkpoint method **always** finds collisions in loops (unlike DP)

### Optimal Theta Selection Formula

**Van Oorschot-Wiener Conjecture** (proven 1999):

When memory is limited to store w distinguished points:
```
θ_optimal ≈ 2.25 / √w

or equivalently:

2^d ≈ 4w / 5.0625
d ≈ log₂(w) + 1.29 bits
```

**Practical Implication**: If your system can store w distinguished points in memory, the optimal DP density is achieved with approximately 2.25 distinguished points per √w operations.

### Communication Overhead in Distributed Settings

For distributed systems with m processors:

**Communication Overhead**: O(m · θ) per unit time

This motivates the need to balance:
- Too small θ: Fast collision detection but large hash table, fills memory
- Too large θ: Low memory usage but slow collision detection, more communication

### Excess Iteration (Wasted Work) Formula

**Proportion exceeding k times average**:
```
P(iterations > k·E[iterations]) = (1 - θ)^k ≈ e^(-k·ln(1/θ))
```

For k=20 and typical θ values:
```
P(length > 20·E) ≈ (1-θ)^20 ≈ e^(-20)
```

This extremely small probability (<10^(-9)) shows distinguished points rarely create extreme path lengths.

---

## Optimal Parameter Selection

### Step 1: Determine RAM Budget

**Input**: Available memory M (in gigabytes)

**Calculation**:
```
w = M_available_bytes / (element_size_bytes)

For secp256k1 ECDLP:
- Each point requires ~32 bytes (x-coordinate + metadata)
- If M = 8GB: w ≈ 256 million distinguished points
```

### Step 2: Calculate Optimal DP Density (θ)

**Method A: Van Oorschot-Wiener Formula**

```
θ_optimal = 2.25 / √w

For w = 256M points:
θ_optimal ≈ 2.25 / √(256M) ≈ 2.25 / 16K ≈ 1/7125
dpBit = -log₂(θ) ≈ 12.8 → use dpBit = 13
```

**Method B: Balanced Approach** (Practical)

If overhead budget is β (fraction of total work tolerated as DP overhead):

```
nbKangaroo · 2^dpBit / √N < β · √N

dpBit < log₂(β · √N / nbKangaroo)

Example: For N=2^128, nbKangaroo=2^20, allow β=10%:
dpBit < log₂(0.1 · 2^64 / 2^20) = log₂(0.1 · 2^44) ≈ 43.7
```

### Step 3: Estimate Practical Overhead

**Overhead Ratio**:
```
overhead_factor = nbKangaroo · 2^dpBit / √N

For overhead_factor = 0.5:  extra 50% work due to DP overhead
For overhead_factor = 1.0:  extra 100% work (overhead ≈ 26% at optimum)
For overhead_factor = 2.0:  overhead becomes prohibitive
```

**Expected Operations**:
```
total_ops ≈ 2.0 · √N · (1 + overhead_factor)

Where 2.0 is the basic kangaroo constant, adjusted for multi-kangaroo variants
```

### Step 4: Memory-Time Trade-off Selection

**Given** constraints:
- Search interval: [L, L+N)
- Available memory: w elements
- Number of parallel kangaroos: m

**Choose** dpBit to minimize:
```
Cost = T + α·S

Where:
- T = total time (operations)
- S = total storage (elements)
- α = cost coefficient (operation/memory ratio)
```

**Iterative Process**:

```python
# Pseudocode for optimal dpBit selection
def find_optimal_dpbit(N, w, nbKangaroo, alpha=1.0):
    best_cost = float('inf')
    best_dpbit = 8

    for dpbit in range(8, 32):
        theta = 1.0 / (2 ** dpbit)
        storage = 2 * math.sqrt(N / theta)

        if storage > w:
            continue  # Exceeds memory budget

        overhead = nbKangaroo * (2 ** dpbit) / math.sqrt(N)
        operations = 2.0 * math.sqrt(N) * (1 + overhead)

        cost = operations + alpha * storage

        if cost < best_cost:
            best_cost = cost
            best_dpbit = dpbit

    return best_dpbit
```

### Practical Guidelines

| Scenario | RAM | dpBit Range | Notes |
|----------|-----|------------|-------|
| Single GPU (8GB) | 8GB | 12-16 | Balance memory and ops |
| Multiple GPUs (32GB) | 32GB | 10-14 | More storage, reduce DP overhead |
| Very Large Search (2^256) | 256GB | 8-12 | Sparse DP, many operations |
| Small Interval (2^64) | 2GB | 16-20 | Dense DP practical |
| Distributed (cloud) | ~1TB | 6-10 | Prioritize lower overhead |

---

## Checkpoint Variants

### Alternative 1: Minimum Value Checkpoint Method

**Algorithm**: Store minimum value every N steps instead of distinguished points

**Advantages**:
- Uniform distribution of checkpoints (N-step intervals)
- Never falls into loopless cycles (always finds collision in loops)
- Slightly better performance: 1.295√N vs 1.309√N for DP
- Simpler implementation (no leading-bit detection needed)

**Disadvantages**:
- Fixed N parameter requires careful tuning
- Cannot parallelize efficiently with rho method (but works with lambda)
- Memory management more complex

**Implementation**:
```
for each kangaroo i:
    j = 0
    position = start_position[i]
    minimum_value = position

    while true:
        j += 1
        position = jump(position)

        if position < minimum_value:
            minimum_value = position

        if j mod N == 0:
            # Store minimum_value in hash table
            # Check for collision with other kangaroos
            check_collision(minimum_value)
            # Continue with next N steps
```

**Choosing N Parameter**:
```
Optimal N ≈ √(N / w)

Where:
- N = total search interval size
- w = number of stored checkpoints available in memory

For N=2^128, w=2^27 (128M entries):
N_param ≈ √(2^128 / 2^27) = 2^50.5 ≈ 1.4 * 10^15 steps/checkpoint
```

### Alternative 2: Hybrid Checkpoint-DP Method

Combines benefits of both approaches:

**Strategy**:
1. Store minimum values at large intervals (N_coarse)
2. Within each interval, track distinguished points (smaller dpBit_fine)
3. Collision detection happens at finer DP level
4. Never loses kangaroos to loopless cycles

**Storage Formula**:
```
S_hybrid = S_coarse + S_fine
         = √N/N_coarse + (√N/θ_fine) · (N_coarse/√N)
         = √N/N_coarse + √N·N_coarse/θ_fine
```

**Trade-off**: Minimize: T + α·S_hybrid

### Alternative 3: Bloom Filter Variant

**Concept**: Use probabilistic data structure instead of hash table

**Advantages**:
- Dramatically reduced memory (1-2 bits per point)
- No collision lookup: points either possibly seen or definitely not seen
- Natural parallelization

**Disadvantages**:
- False positives require verification
- Cannot distinguish between tame and wild paths

**Formula**:
```
Bloom filter size = w_bits
False positive rate = (1 - e^(-m·w/size))^m
```

### Alternative 4: Nested DP Levels

Multiple DP thresholds with increasing selectivity:

**Levels**:
1. **Coarse level**: dpBit=8 (sparse, low memory, frequent detection)
2. **Fine level**: dpBit=14 (denser, moderate memory)
3. **Verification level**: exact match checking

**Advantage**: Adaptive overhead based on collision proximity

---

## Reducing Wasted Work

### 1. Wasted Work Sources

**Type A: Excess Iterations Before DP**
- Path continues after collision, before reaching distinguished point
- Expected excess: E[excess] = (1-θ)/(2θ) ≈ 1/(2θ) for small θ
- For dpBit=13 (θ ≈ 1/8192): expected ~4096 excess steps

**Type B: Cycle-Trapping** (DP method only)
- Random walk enters cycle with no distinguished point
- Probability P(trapped) ≈ 1 - e^(-√N · θ) for reasonably sized N
- Solution: Restart with new seed (loses all progress)

**Type C: Duplicate Detections**
- Same collision detected multiple times in parallel settings
- When continuation method used (starting from DP instead of fresh seed)
- Probability P(duplicate | m processors) ≈ 1 - e^(-m²/2N)

### 2. Quantifying Excess Iterations

**For Distinguished Points with dpBit=d**:

```
E[steps to DP after collision] = (2^d - 1) / 2 ≈ 2^(d-1)

For dpBit=13:
E[excess] ≈ 2^12 ≈ 4096 steps

For dpBit=8:
E[excess] ≈ 2^7 ≈ 128 steps
```

**Total Cost of One Collision Detection**:
```
Cost = E[steps until collision] + E[excess steps to DP]
     ≈ 2√N + 2^(dpBit-1)

With dpBit=13:
Cost ≈ 2√N + 4096

For N=2^128: Cost ≈ 2·2^64 + 4096 ≈ 2·2^64 (4096 is negligible)
```

### 3. Minimizing Excess Work

#### Strategy A: Adjust dpBit to Trade Excess for Overhead

**Higher dpBit**:
- Excess steps: 2^(dpBit-1) increases exponentially
- But happens less frequently
- Total effect small if dpBit ≥ log₂(√N) - k for reasonable k

**Lower dpBit**:
- Excess steps decrease
- But DP detection happens sooner (better)
- Trades against higher memory consumption

**Optimal Trade-off**:
```
d_optimal ≈ log₂(√(N·α / nbKangaroo))

Where α = communication overhead coefficient
```

#### Strategy B: Early Termination

**Idea**: Declare collision "probably detected" before reaching DP if paths converge

**Implementation**:
```
when |position_tame - position_wild| < 2^d:
    - Paths are within DP detection distance
    - Verify exact collision at this point
    - 50% savings on excess work if detected early
```

#### Strategy C: Use Checkpoint Method

**Why**: Eliminates cycle-trapping entirely

**Comparison**:
```
DP method:
- Excess steps: E[excess] ≈ 2^(dpBit-1) per collision
- Cycle-trap cost: variable, can be entire restart
- Total expected: 2√N + occasional full restarts

Checkpoint method:
- Excess steps: E[excess] ≈ N/2 (fixed, predictable)
- Never cycle-traps
- Total expected: 2√N + N/2 = always predictable
```

**When Checkpoint is Better**:
```
Use checkpoint when N/2 < E[excess from cycles]
or when parallelization requires deterministic behavior
```

### 4. Batch Processing and Buffering

**Strategy**: Collect multiple DP/checkpoints before communication/verification

**Benefit**:
```
- Single cache miss costs ~500 cycles
- Batching 16-64 points amortizes to ~30 cycles per point
- Overhead reduction: up to 30%
```

**Implementation**:
```
Buffer size B ≈ L1_cache_size / (element_size)
For typical: B ≈ 32KB / 32B ≈ 1K entries

Expected work reduction ≈ log(B) / B ≈ 10%
```

### 5. Loop Detection and Restart Strategy

**Probability of Entering Loopless Cycle**:
```
P(loopless) ≈ 1 - e^(-√(N·θ) / √(order))

For ECDLP with proper grouping: extremely rare
But non-zero, affecting distributed runs
```

**Mitigation**:
1. **Detection**: Monitor for convergence stall
2. **Restart**: Switch to new random seed
3. **Memory**: Save current state before restart to avoid redundant work

**Maximum Path Length** (with high confidence):
```
L_max ≈ √N + c·√(√N·√w)

For N=2^128, w=2^27:
L_max ≈ 2^64 + c·2^45 (still dwarfs 2^64)
```

---

## Implementation Considerations

### 1. Implementing DP Detection in Code

#### Pseudocode Structure

```c
struct Point {
    uint8_t x[32];      // x-coordinate
    uint64_t step_count;
    uint32_t kangaroo_id;
};

// Distinguished Point detection
bool is_distinguished_point(const Point* p, int dpBit) {
    // Check if first dpBit bits are zero
    // For secp256k1: check first dpBit bits of p->x[0]

    int full_bytes = dpBit / 8;
    int remaining_bits = dpBit % 8;

    // Check full bytes
    for (int i = 0; i < full_bytes; i++) {
        if (p->x[i] != 0) return false;
    }

    // Check remaining bits
    if (remaining_bits > 0) {
        uint8_t mask = (0xFF << (8 - remaining_bits));
        if ((p->x[full_bytes] & mask) != 0) return false;
    }

    return true;
}

// Main collision detection loop
void kangaroo_step(Point* tame, Point* wild, HashTable* table, int dpBit) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        // Tame kangaroo jump
        tame->x = jump_function(tame->x, TAME_JUMP);
        tame->step_count++;

        if (is_distinguished_point(tame, dpBit)) {
            if (hash_table_lookup(table, tame->x)) {
                // Collision detected!
                return COLLISION_FOUND;
            }
            hash_table_insert(table, tame->x, tame->step_count);
        }

        // Wild kangaroo jump
        wild->x = jump_function(wild->x, WILD_JUMP);
        wild->step_count++;

        if (is_distinguished_point(wild, dpBit)) {
            if (hash_table_lookup(table, wild->x)) {
                // Collision detected!
                return COLLISION_FOUND;
            }
            hash_table_insert(table, wild->x, wild->step_count);
        }
    }
}
```

#### Performance Optimization

**1. Bitwise DP Check**:
```c
// Instead of loop, use bit comparison
uint64_t mask = (dpBit >= 64) ? 0xFFFFFFFFFFFFFFFF
                              : ((1ULL << (64 - dpBit)) - 1);

bool is_dp = (x[0] & mask) == 0;
```

**2. Hash Table Optimization**:
```c
// Use rolling hash for x-coordinate
uint64_t hash(const uint8_t* x) {
    // Fast hash using first 8 bytes
    return *(uint64_t*)x;
}

// Instead of full lookup, use hash + verification
size_t bucket = hash(x) & (table_size - 1);
```

**3. SIMD Batch Processing**:
```c
// Process 4 points in parallel (AVX2)
__m256i x0, x1, x2, x3;
__m256i zero = _mm256_setzero_si256();

// Load 4 x-coordinates
// ... SIMD comparison logic ...
// Store DP flags to batch buffer
```

### 2. Adapting to GPU/CUDA Implementation

#### Memory Layout

```cuda
struct GPUPoint {
    uint32_t x[8];      // secp256k1 x-coord (256-bit)
    uint32_t y[8];      // optional y-coord
    uint64_t steps;     // steps taken
};

// Global memory: ~10-20% for active points
// Shared memory: DP table cache
// Constant memory: jump table (precomputed)

__global__ void kangaroo_kernel(GPUPoint* tame, GPUPoint* wild,
                                uint32_t dpBit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread manages one kangaroo
    GPUPoint my_tame = tame[idx];
    GPUPoint my_wild = wild[idx];

    for (int step = 0; step < STEPS_PER_KERNEL; step++) {
        // Tame jump
        my_tame.x = ec_jump(my_tame.x, TAME_JUMP);
        my_tame.steps++;

        if (is_distinguished_point_gpu(my_tame.x, dpBit)) {
            // Add to global hash table
            // Synchronization needed
        }

        // Similar for wild
        // ...
    }

    tame[idx] = my_tame;
    wild[idx] = my_wild;
}
```

#### DP Detection in CUDA

```cuda
__device__ bool is_distinguished_gpu(const uint32_t* x, int dpBit) {
    // First dpBit bits must be zero

    if (dpBit <= 32) {
        // Check first word only
        uint32_t mask = (dpBit == 32) ? 0xFFFFFFFF : ((1U << (32 - dpBit)) - 1);
        return (x[0] & mask) == 0;
    } else {
        // Check multiple words
        for (int i = 0; i < dpBit / 32; i++) {
            if (x[i] != 0) return false;
        }

        int remaining = dpBit % 32;
        if (remaining > 0) {
            uint32_t mask = (1U << (32 - remaining)) - 1;
            if ((x[dpBit / 32] & mask) != 0) return false;
        }

        return true;
    }
}
```

#### Distributed Hash Table (Multi-GPU)

```cuda
// Central server
class DistributedHashTable {
    // GPU 0: tame kangaroo DP storage
    // GPU 1: wild kangaroo DP storage
    // GPU N: merged results

    void synchronize() {
        // Collect DP points from all GPUs
        // Check tame vs wild collisions
        // Report to host
    }
};
```

### 3. dpBit Auto-Tuning

```python
def calculate_optimal_dpbit(interval_size, memory_bytes, num_kangaroos):
    """
    Auto-calculate optimal dpBit for given constraints.

    Args:
        interval_size: size of search interval (N)
        memory_bytes: available RAM for DP table
        num_kangaroos: number of parallel kangaroos

    Returns:
        optimal_dpbit: recommended dpBit value
        expected_overhead: percentage overhead
    """

    import math

    element_size = 32  # bytes per x-coordinate + metadata
    max_dp_entries = memory_bytes // element_size

    sqrt_n = math.sqrt(interval_size)

    # Van Oorschot-Wiener formula
    theta_optimal = 2.25 / math.sqrt(max_dp_entries)
    dpbit_vow = -math.log2(theta_optimal)

    # Constraint: overhead < 50%
    max_overhead = 0.5
    dpbit_max = math.log2(max_overhead * sqrt_n / num_kangaroos)

    # Choose conservative value
    dpbit = int(min(dpbit_vow, dpbit_max))

    # Verify practical constraints
    theta = 1.0 / (2 ** dpbit)
    expected_storage = 2 * sqrt_n / math.sqrt(theta)

    if expected_storage > max_dp_entries * element_size:
        # Adjust upward
        dpbit += 1

    overhead = num_kangaroos * (2 ** dpbit) / sqrt_n

    return dpbit, overhead

# Example usage
dpbit, overhead = calculate_optimal_dpbit(
    interval_size=2**128,
    memory_bytes=8 * 1024**3,  # 8GB
    num_kangaroos=256
)
print(f"Recommended dpBit: {dpbit}")
print(f"Expected overhead: {overhead:.1%}")
```

### 4. Work File Management

**Saving State** (with DP overhead accounting):

```python
def save_work_file(filename, kangaroos, hash_table, dpBit, work_done):
    """
    Save work file for resumption.

    Note: If resuming with different dpBit or hardware,
    DP overhead will cause some lost work (~5-15%).
    """

    state = {
        'kangaroos': kangaroos,
        'hash_table': hash_table,
        'dpBit': dpBit,
        'work_done': work_done,
        'timestamp': time.time()
    }

    # Using -ws flag: save kangaroos to minimize overhead
    if use_kangaroo_save:
        state['kangaroos_preserved'] = True

    with open(filename, 'wb') as f:
        pickle.dump(state, f)

def resume_work_file(filename, new_dpBit=None):
    """
    Resume from saved state.

    Cost of changing dpBit:
    - Recalculate DP detection (small cost)
    - May lose ~2-8% work from DP overhead change
    - Recalculate hash table structure
    """

    with open(filename, 'rb') as f:
        state = pickle.load(f)

    if new_dpBit and new_dpBit != state['dpBit']:
        # Warn about overhead
        overhead_cost = estimate_overhead_change(
            state['dpBit'], new_dpBit, state['work_done']
        )
        print(f"Warning: Changing dpBit costs ~{overhead_cost:.1%} efficiency")
```

### 5. Parallelization Strategy

**Key Principle**: Ensure gcd(U, V) = 1 where U, V = tame and wild herds

```python
class KangarooHerd:
    def __init__(self, size, herd_type='tame'):
        self.size = size
        self.herd_type = herd_type
        self.kangaroos = [...]

    def parallelize(self, num_processors):
        """
        Create parallel herds with coprime sizes to avoid
        "useless collisions" (collisions between same herd).

        Returns: (tame_herd, wild_herd) both of size num_processors
        """

        import math

        # Ensure coprime herd sizes
        tame_count = num_processors
        wild_count = num_processors

        # Check: gcd(tame_count, wild_count) must be 1
        while math.gcd(tame_count, wild_count) != 1:
            wild_count += 1

        # Distribute kangaroos
        tame_herd = self.create_herd(tame_count, 'tame')
        wild_herd = self.create_herd(wild_count, 'wild')

        return tame_herd, wild_herd
```

### 6. Configurable Parameters Summary

```ini
# Example configuration file for Kangaroo solver

[interval]
# Search range size N
bit_length = 128
start = 0x0

[algorithm]
# Distinguished point method
collision_method = distinguished_points
# or: collision_method = checkpoint (using Wang-Zhang method)

[optimization]
# DP density parameter
dpBit = auto              # auto-calculate, or specify 8-20
dpBit_min = 8            # minimum value
dpBit_max = 20           # maximum value

# Kangaroo parameters
kangaroo_count = 256      # per GPU
kangaroo_type = 4         # 2=basic, 3=three, 4=four, 7=seven

[hardware]
# GPU configuration
gpu_count = 1
threads_per_gpu = 2048

[memory]
# RAM allocation
max_ram_gb = 8
hash_table_percent = 80   # % of RAM for DP table

[performance]
# Overhead tolerance
max_overhead_percent = 25
batch_size = 1024         # DP checks per kernel invocation

[output]
# Work files
save_interval_hours = 6
save_with_kangaroos = true  # minimize overhead on resume
```

---

## References

### Primary Academic Papers

1. **van Oorschot, P. C., & Wiener, M. J.** (1999)
   "Parallel Collision Search with Cryptanalytic Applications"
   *Journal of Cryptology*, 12(1), 1-28.
   - Foundational work on distinguished points and parallel collision search
   - Introduces optimal theta selection framework

2. **Wang, P., & Zhang, F.** (2012)
   "An Efficient Collision Detection Method for Computing Discrete Logarithms with Pollard's Rho"
   *Journal of Applied Mathematics*, Article 635909
   - Presents minimum-value checkpoint method as alternative to DP
   - Demonstrates 1% performance improvement
   - Hindawi Publishing: https://www.hindawi.com/journals/jam/2012/635909/

3. **Fowler, A., & Galbraith, S. D.** (2015)
   "Kangaroo Methods for Solving the Interval Discrete Logarithm Problem"
   *arXiv preprint 1501.07019*
   - Advances multi-kangaroo methods
   - Achieves 1.715√N complexity with seven kangaroos
   - Analysis of DP collision detection in advanced variants

4. **Galbraith, S. D., Pollard, J. M., & Ruprai, R. S.** (2012)
   - Three and four kangaroo methods reducing to 1.818√N and 1.714√N
   - Trade-off analysis between variants

### Time-Space Trade-off Theory

5. **Hellman, M. E.** (1980)
   "A Cryptanalytic Time-Memory Trade-Off"
   *IEEE Transactions on Information Theory*, 26(4), 401-406.
   - Original time-memory trade-off concept
   - Precomputation tables foundation

6. **Oechslin, P.** (2003)
   "Making a Faster Cryptanalytic Time-Memory Trade-Off"
   *Crypto 2003*, LNCS 2729, pp. 617-630.
   - Rainbow tables improving Hellman approach
   - Addresses variable chain length overhead

7. **Dinur, I.** (2020)
   "Optimal Time-Space Trade-offs for Sorting"
   *Eurocrypt 2020*
   - Modern optimal T²S = Θ(K²N) bounds
   - Quantum extensions: T³S ≥ Ω(K³N)

### Distinguished Points and Variants

8. **Hong, J., Jeong, K. C., Kwon, E. Y., Lee, I.-S., & Ma, D.** (2008)
   "Variants of the Distinguished Point Method for Cryptanalytic Time Memory Trade-Offs"
   *Cryptology ePrint Archive*, Report 2008/054.
   - Multiple DP variants analysis
   - Hybrid methods combining multiple approaches

9. **Kang, H., & Yi, O.** (2010)
   "On Distinguished Points Method to Implement a Parallel Collision Search Attack on ECDLP"
   *Security Technology, Disaster Recovery and Business Continuity*, CCIS 122, pp. 51-60.
   - Practical ECDLP implementation strategies
   - Optimal theta determination for specific platforms

### Implementation and Practical Guides

10. **Pollard's Kangaroo Algorithm - Wikipedia**
    https://en.wikipedia.org/wiki/Pollard's_kangaroo_algorithm
    - Algorithm overview and basic complexity analysis

11. **JeanLucPons/Kangaroo - GitHub Repository**
    https://github.com/JeanLucPons/Kangaroo
    - Production-grade implementation for secp256k1
    - Real-world parameter tuning and optimization
    - Extensive documentation on dpBit selection

12. **Kangaroos in Side-Channel Attacks**
    *IACR Cryptology ePrint Archive*, Report 2014/565
    - Security considerations for parallel implementations
    - Side-channel resistant parameter selection

### Related Problem Variants

13. **Bai, S., et al.**
    "On the Efficiency of Pollard's Rho Method for Discrete Logarithms"
    - Complexity analysis and constant factors
    - Practical performance measurements

14. **Galbraith, S. D.** (2012)
    "Mathematics of Public Key Cryptography"
    Cambridge University Press, Chapter on ECDLP algorithms
    - Comprehensive mathematical treatment
    - Convergence analysis for distinguished point methods

---

## Appendix: Quick Reference

### Memory-Time Trade-off Quick Lookup

| N (bits) | Ideal dpBit (8GB) | Ideal dpBit (32GB) | Approx Ops | RAM Usage |
|----------|-------------------|--------------------|-----------|-----------|
| 64 | 18-20 | 16-18 | 2^32 ops | 100MB-1GB |
| 80 | 16-18 | 14-16 | 2^40 ops | 500MB-5GB |
| 96 | 14-16 | 12-14 | 2^48 ops | 2GB-8GB |
| 112 | 12-14 | 10-12 | 2^56 ops | 4GB-8GB |
| 128 | 10-12 | 8-10 | 2^64 ops | 8GB+ |
| 160 | 8-10 | 6-8 | 2^80 ops | 32GB+ |
| 256 | 4-6 | 2-4 | 2^128 ops | 256GB+ |

### Formula Summary Sheet

```
Basic Kangaroo:
  E[ops] = 2√N
  E[storage] = 2√(N/θ)

DP Density:
  θ = 1/2^dpBit
  Overhead ratio = nbKangaroo · 2^dpBit / √N

Optimal Theta (limited memory):
  θ_optimal ≈ 2.25 / √w
  dpBit ≈ log₂(√w / 2.25)

Expected Excess Steps:
  E[excess | DP] ≈ 2^(dpBit-1)
  E[excess | checkpoint] = N/2

Multi-kangaroo constants:
  2-kangaroo: 2.00√N
  3-kangaroo: 1.818√N
  4-kangaroo: 1.714√N
  7-kangaroo: 1.715√N

Time-Space Trade-off:
  T² · S ≈ K² · N (van Oorschot-Wiener)
  Total Cost ≈ E[ops] + α·E[storage]
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-01
**Subject**: Collision Detection Optimization for Pollard's Kangaroo ECDLP Solver
**Scope**: Comprehensive technical reference for systematic time-space trade-offs

This document synthesizes research from the field's leading papers and implementations to provide actionable guidance for optimizing collision detection in discrete logarithm solvers.
