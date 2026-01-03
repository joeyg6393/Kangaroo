# Multi-Kangaroo Patterns: Advanced Configurations for Pollard's Kangaroo ECDLP Solver

## Overview

The Multi-Kangaroo Patterns upgrade represents a significant theoretical and practical advancement in Pollard's kangaroo method for solving the interval discrete logarithm problem (IDLP). Rather than using the standard two-kangaroo configuration (one tame, one wild), this upgrade employs multiple coordinated kangaroos (3, 4, 5, or 7) to achieve substantially improved constant factors in the expected running time complexity.

The foundational insight is that by strategically deploying multiple kangaroos with different starting positions and walk trajectories, the algorithm can achieve collision detection with fewer total group operations. The improvements over the standard Pollard's original method (3.3√N) and Van Oorschot-Wiener method (2√N) are substantial:

- **Three Kangaroo Method**: (1.818 + o(1))√N group operations
- **Four Kangaroo Method**: (1.714 + o(1))√N group operations
- **Five Kangaroo Method**: (1.737 + o(1))√N group operations
- **Seven Kangaroo Method**: (1.7195 + o(1))√N ± O(1) group operations

This represents a roughly 47% improvement over the original Pollard method and a 14% improvement over Van Oorschot-Wiener, with the four and seven kangaroo methods providing near-optimal performance.

## Key Concepts

### The Interval Discrete Logarithm Problem

The fundamental problem is defined as:
- Given: g, h ∈ G (a group), and N ∈ ℕ
- Find: z where g^z = h and 0 ≤ z < N

Unlike the standard discrete logarithm problem requiring a full group search, the interval constraint allows for efficient targeted algorithms.

### Kangaroo Walks and Pseudorandom Walks

Each kangaroo performs a pseudorandom walk through the group G, where:
- The next position is deterministically computed from the current position
- Jump sizes are determined by a hash function F applied to the current group element
- The jumps appear random but are fully deterministic and reproducible
- Two kangaroos following the same path will continue together (they "merge")

### Collision Detection via Markov Chains

The theoretical foundation rests on Markov chain analysis:
- Each kangaroo's path forms a Markov chain with pseudorandom properties
- The algorithm leverages the Birthday Paradox: with √N expected collisions, two independent random walks will intersect
- Multiple kangaroos increase the collision probability space - more pairs of kangaroos means more opportunities for collisions
- When collisions occur between different kangaroo types with known starting positions, the discrete logarithm can be directly computed

### Multi-Kangaroo Advantage

With k kangaroos in coordinated walks:
- Create k(k-1)/2 collision pairs instead of just 1 (tame-wild pair)
- Reduce expected collision time by optimizing starting positions and walk structures
- Trade increased computational coordination complexity for reduced total operations

## Optimal Configurations

### Three Kangaroo Method (1.818√N)

**Kangaroo Types and Starting Positions:**
- **T (Tame)**: Starts at position 3N/10
- **W1 (Wild1)**: Starts at position z - N/2
- **W2 (Wild2)**: Starts at position N/2 - z

**Walk Representations:**
- Tame: Elements of form g^q
- Wild1: Elements of form g^(ph) where h = g^z
- Wild2: Elements of form g^(rh^(-1)) = g^(r-z)

**Collision Solving:**
- If T collides with W1: x = g^q = g^p → z = p - q
- If T collides with W2: x = g^q = g^r·h^(-1) → z = r - q
- If W1 collides with W2: x = g^p·h = g^r·h^(-1) → z = (r-p)/2

**Distance Functions:**
- d_T,W1(z) = 2N/5 - z
- d_T,W2(z) = N/5 - z
- d_W1,W2(z) = 2z - N

The expected running time accounts for the three independent collision opportunities and the probabilistic expected collision times between each pair.

### Four Kangaroo Method (1.714√N)

**Evolution from Three Kangaroo:**
The four kangaroo method extends the three-kangaroo approach with optimized starting positions and walk structures to reduce the expected running time from 1.818√N to 1.714√N.

**Key Improvements:**
- Four instead of three collision pairs
- Refined starting position calculations that reduce average collision distances
- Galbraith, Pollard, and Ruprai's analysis shows this represents a near-optimal point in the efficiency curve
- The method was the fastest non-parallelized kangaroo variant prior to work on 5+ kangaroos

**Heuristic Running Time:**
Empirical and heuristic analysis shows (1.714 + o(1))√N group operations with memory requirement of O(log N).

### Five Kangaroo Method (1.737√N)

**Theoretical Question Addressed:**
"Are five kangaroos worse than three?" This question motivated the analysis of intermediate configurations.

**Performance Finding:**
Surprisingly, five kangaroos is WORSE than three (1.737√N vs 1.818√N), despite offering ten collision pairs versus three. This demonstrates that:
- Simply increasing kangaroo count doesn't automatically improve performance
- The optimal number of collision pairs and their expected distances matter more than raw pair count
- Four kangaroos represent a local optimum before the jump to seven

**Implication:**
Five kangaroos provide worse constant factors, making three or four preferable for practical implementation unless combined with other optimizations.

### Seven Kangaroo Method (1.7195√N)

**Breaking the Four-Kangaroo Barrier:**
Prior to this work, any method using more than four kangaroos required at least 2√N operations. The seven kangaroo method represents a breakthrough, nearly matching the four-kangaroo efficiency while using substantially more coordinated walks.

**Configuration Details:**
- Seven coordinated kangaroos with carefully optimized starting positions
- 21 unique collision pairs (7 × 6 / 2)
- Expected running time: (1.7195 + o(1))√N ± O(1) group operations
- The ± O(1) term reflects higher-order variance considerations

**Why Seven Works:**
The seven kangaroo configuration strikes an optimal balance between:
- Multiple collision opportunities (21 pairs)
- Expected distances between starting positions
- Walk intersection probabilities
- Coordination overhead

**Practical Implications:**
- Only 0.5% faster than four kangaroos (1.7195 vs 1.714)
- Requires 75% more kangaroo management/coordination
- May not be worth additional complexity for typical implementations
- More valuable for massively parallel implementations where coordination cost is amortized

## Expected Benefits

### Constant Factor Improvements

The progression shows diminishing returns:

| Configuration | Constant c in c√N | Improvement vs Pollard | Improvement vs Van Oorschot-Wiener |
|---|---|---|---|
| Pollard (1978) | 3.300 | - | 65% slower |
| Van Oorschot-Wiener | 2.000 | 39% faster | - |
| Ruprai (Three) | 1.818 | 45% faster | 9% faster |
| Galbraith-Pollard-Ruprai (Four) | 1.714 | 48% faster | 14% faster |
| Five Kangaroo | 1.737 | 47% faster | 13% faster |
| Seven Kangaroo | 1.7195 | 48% faster | 14% faster |

### Real-World Time Savings

For a 256-bit ECDLP problem:
- **Pollard Method**: ~2^128 operations expected
- **Four Kangaroo**: ~2^127 × 1.714 operations (1.5% less than 2^128)
- **Seven Kangaroo**: ~2^127 × 1.7195 operations (1.4% less than 2^128)

For a 128-bit problem (N = 2^128):
- Pollard: ~3.3 × 2^64 ≈ 5.9 × 10^19 operations
- Four Kangaroo: ~1.714 × 2^64 ≈ 3.1 × 10^19 operations
- Seven Kangaroo: ~1.7195 × 2^64 ≈ 3.1 × 10^19 operations

The constant factor advantage translates directly to wall-clock time savings proportional to the constant factor reduction.

### Memory Efficiency

All kangaroo variants maintain O(log N) memory complexity via distinguished points:
- Only points with x-coordinates starting with dpBit zero bits are stored
- Multiple kangaroos share the same distinguished points hash table
- Memory scales with number of distinguished points, not number of kangaroos
- Typical memory usage: 10-20 GB for 256-bit problems with modern distinguished point parameters

## Implementation Considerations

### GPU Kernel Architecture for Multiple Kangaroos

#### Single-Kangaroo Kernel (Baseline)

```c
// Basic single kangaroo step in GPU kernel
__global__ void kangaroo_step_single(
    Point *positions,      // Current position of kangaroo
    uint64_t *exponents,   // Current exponent in walk
    uint32_t *dp_bits,     // Distinguished point threshold
    uint32_t iterations
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    Point p = positions[idx];
    uint64_t exp = exponents[idx];

    for(uint32_t i = 0; i < iterations; i++) {
        uint32_t jump = hash_function(p.x) & JUMP_MASK;
        p = ec_point_add(p, precomputed_jumps[jump]);
        exp += jump_sizes[jump];

        if(is_distinguished_point(p, dp_bits)) {
            store_distinguished_point(p, exp);
        }
    }
}
```

#### Multi-Kangaroo Kernel (Three Kangaroos)

```c
// Multi-kangaroo step with coordinated walks
__global__ void kangaroo_step_multi(
    Point *positions_t,    // Tame kangaroo positions
    Point *positions_w1,   // Wild1 kangaroo positions
    Point *positions_w2,   // Wild2 kangaroo positions
    uint64_t *exp_t,       // Tame exponents
    uint64_t *exp_w1,      // Wild1 exponents
    uint64_t *exp_w2,      // Wild2 exponents
    uint32_t *dp_bits,
    uint32_t iterations
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process tame kangaroo
    Point p_t = positions_t[idx];
    uint64_t e_t = exp_t[idx];
    for(uint32_t i = 0; i < iterations; i++) {
        uint32_t jump = hash_function(p_t.x) & JUMP_MASK;
        p_t = ec_point_add(p_t, precomputed_jumps[jump]);
        e_t += jump_sizes[jump];
        if(is_distinguished_point(p_t, dp_bits)) {
            store_distinguished_point(p_t, e_t, KANGAROO_TAME);
        }
    }
    positions_t[idx] = p_t;
    exp_t[idx] = e_t;

    // Process wild kangaroo 1
    Point p_w1 = positions_w1[idx];
    uint64_t e_w1 = exp_w1[idx];
    for(uint32_t i = 0; i < iterations; i++) {
        uint32_t jump = hash_function(p_w1.x) & JUMP_MASK;
        p_w1 = ec_point_add(p_w1, precomputed_jumps[jump]);
        e_w1 += jump_sizes[jump];
        if(is_distinguished_point(p_w1, dp_bits)) {
            store_distinguished_point(p_w1, e_w1, KANGAROO_WILD1);
        }
    }
    positions_w1[idx] = p_w1;
    exp_w1[idx] = e_w1;

    // Process wild kangaroo 2
    Point p_w2 = positions_w2[idx];
    uint64_t e_w2 = exp_w2[idx];
    for(uint32_t i = 0; i < iterations; i++) {
        uint32_t jump = hash_function(p_w2.x) & JUMP_MASK;
        p_w2 = ec_point_add(p_w2, precomputed_jumps[jump]);
        e_w2 += jump_sizes[jump];
        if(is_distinguished_point(p_w2, dp_bits)) {
            store_distinguished_point(p_w2, e_w2, KANGAROO_WILD2);
        }
    }
    positions_w2[idx] = p_w2;
    exp_w2[idx] = e_w2;
}
```

### Key Implementation Challenges

#### 1. Collision Detection Overhead

With multiple kangaroos:
- Hash table lookups become critical bottleneck
- Each distinguished point write requires hash table access
- With N kangaroos doing 2√N steps, distinguished point collisions occur ~4N times
- Solution: Use lock-free hash tables, batch lookups, or hierarchical detection

#### 2. Starting Position Management

**Three Kangaroo Example:**
```c
// Initialize kangaroo starting positions
__global__ void initialize_kangaroos(
    Point *positions_t, Point *positions_w1, Point *positions_w2,
    uint64_t *exp_t, uint64_t *exp_w1, uint64_t *exp_w2,
    const Point target,  // h = g^z
    uint64_t N
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Tame kangaroo at position 3N/10
    uint64_t exp_start_t = (3 * N) / 10;
    positions_t[idx] = ec_point_multiply(g, exp_start_t);
    exp_t[idx] = exp_start_t;

    // Wild1 at position z - N/2 (z unknown, so use symbolic)
    // In practice, parameterized relative to z
    positions_w1[idx] = ec_point_add(target, ec_point_multiply(g, -N/2));
    exp_w1[idx] = -N/2;  // Relative to unknown z

    // Wild2 at position N/2 - z
    positions_w2[idx] = ec_point_add(ec_point_multiply(g, N/2),
                                      ec_point_negate(target));
    exp_w2[idx] = N/2;   // Relative to -z
}
```

#### 3. Exponent Tracking in Multiple Walks

Multi-kangaroo algorithms require tracking exponents for collision solving:

```c
// Collision detection and resolution
__host__ bool check_collision_and_solve(
    const Point &p,           // Distinguished point reached
    const uint64_t exp_finder, // Exponent of finding kangaroo
    int kangaroo_type,        // TAME, WILD1, or WILD2
    const Point &target_h,    // The h we're solving for
    uint64_t &solution_z      // Output: the discrete log
) {
    // Check if this point was reached by another kangaroo type
    auto it = distinguished_points_table.find(p);
    if(it != distinguished_points_table.end()) {
        for(auto &stored : it->second) {
            if(stored.kangaroo_type != kangaroo_type) {
                // Collision found!

                // Solve for z based on collision types
                if(kangaroo_type == TAME && stored.kangaroo_type == WILD1) {
                    // T = g^q, W1 = h·g^p → z = p - q
                    solution_z = stored.exponent - exp_finder;
                } else if(kangaroo_type == TAME && stored.kangaroo_type == WILD2) {
                    // T = g^q, W2 = h^(-1)·g^r → z = r - q
                    solution_z = stored.exponent - exp_finder;
                } else if(kangaroo_type == WILD1 && stored.kangaroo_type == WILD2) {
                    // W1 = h·g^p, W2 = h^(-1)·g^r → z = (r - p) / 2
                    solution_z = (stored.exponent - exp_finder) / 2;
                }

                return true;
            }
        }
    }
    return false;
}
```

#### 4. Load Balancing Across Kangaroos

Challenge: Different kangaroos may have different expected step distributions
- Tame kangaroos: Uniform distribution across [0, N]
- Wild kangaroos: Clustered around target value z
- Solution: Use work stealing or dynamic scheduling

```c
// Work distribution for four kangaroos
#define NUM_KANGAROOS 4
#define KANGAROO_BLOCKS_PER_GPU 1024

__global__ void kangaroo_dispatcher(
    KangarooState *kangaroos[NUM_KANGAROOS],
    uint32_t blocks_per_kangaroo
) {
    for(int k = 0; k < NUM_KANGAROOS; k++) {
        launch_kangaroo_kernel<<<blocks_per_kangaroo, 256>>>(
            kangaroos[k], k
        );
    }
}
```

### Optimization Strategies

1. **Precomputed Jump Tables**: Cache all 2^k possible jumps to avoid recomputation
2. **Distinguished Point Filtering**: Use variable dp_bits (8-32) to balance memory vs. detection time
3. **Batch Distinguished Point Lookup**: Accumulate distinguished points before global synchronization
4. **Thread-Level Parallelism**: Run multiple independent walks per thread for better occupancy
5. **Persistent Kernels**: Keep kangaroo computations persistent on GPU without CPU-GPU round trips

## References

### Primary Source
- Fowler, A., & Galbraith, S. D. (2015). "Kangaroo Methods for Solving the Interval Discrete Logarithm Problem." Retrieved from https://arxiv.org/abs/1501.07019

### Foundational Work
- Pollard, J. M. (1978). "Kangaroos, Monopoly and Discrete Logarithms." University of East Anglia preprint.
- Van Oorschot, P. C., & Wiener, M. J. (1999). "Parallel collision search with cryptanalytic applications." *Journal of Cryptology*, 12(1), 1-28.
- Galbraith, S. D., Pollard, J. M., & Ruprai, R. S. (2010). "Computing discrete logarithms in an interval." https://eprint.iacr.org/2010/617.pdf

### Implementation References
- [JeanLucPons/Kangaroo](https://github.com/JeanLucPons/Kangaroo) - GPU-accelerated kangaroo for SECP256K1
- [Kangaroo-256-bit](https://github.com/mikorist/Kangaroo-256-bit) - CUDA-optimized variant with multi-GPU support

### Related Papers
- [Computing Discrete Logarithms with the Parallelized Kangaroo Method](https://www.sciencedirect.com/science/article/pii/S0166218X02005905)
- [How Long Does it Take to Catch a Wild Kangaroo?](https://arxiv.org/pdf/0812.0789) - Markov chain analysis by Montenegro et al.
- [Using Equivalence Classes to Accelerate Solving the Discrete Logarithm Problem in a Short Interval](https://eprint.iacr.org/2010/615)

---

**Document Date**: January 2026
**Status**: Comprehensive technical reference for multi-kangaroo pattern implementations
