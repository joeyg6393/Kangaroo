# Walk/Step-Function Tuning Using Collision Bounds: Pollard's Kangaroo ECDLP Solver

## 1. Overview

### Why Step Function Design Matters for Kangaroo Walks

The efficiency of Pollard's kangaroo algorithm for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP) depends critically on the design of the step function that controls how the algorithm's pseudorandom walks traverse the group. The walk generates a sequence of group elements according to a deterministic but seemingly random iteration function, and the speed at which collisions occur between tame and wild kangaroo walks determines the overall runtime.

The step function acts as the bridge between the theoretical expectation of O(√N) group operations and actual practical performance. A well-designed step function:

- **Approximates random walk behavior**: Ensures the walk behaves statistically like a random walk on the group structure
- **Minimizes walk variance**: Reduces the variance in step sizes, which improves convergence properties
- **Balances collision detection**: Optimizes the rate at which collisions are detected through partition-based collision criteria
- **Enables efficient parallelization**: Maintains independence between multiple walks in parallel configurations

The choice between simple binary partitioning (3 sets) versus refined partitioning (r ≈ 20 sets) represents a fundamental trade-off between simplicity and performance. Research by Teske and others has demonstrated that refining the partition structure can yield speed-ups of 20% or more in actual execution time.

### Historical Context

Pollard's original kangaroo method (1978) used a basic step function with three partition sets. Subsequent improvements by Pollard, Ruprai, and Galbraith extended the kangaroo method to multi-kangaroo variants (three-kangaroo and four-kangaroo methods) achieving expected runtimes of approximately (1.818+o(1))√N and (1.714+o(1))√N group operations respectively. These improvements primarily came from algorithmic refinements (more kangaroos), but step function design remains a critical component for practical implementations.

## 2. Markov Chain Analysis

### Mathematical Modeling of Random Walks

The kangaroo walk can be rigorously modeled as a Markov chain on the cyclic group G of order n. The iteration function F: G → G defines a sequence (y_i) where y_{i+1} = F(y_i), creating a finite-state Markov chain with n states (group elements) and deterministic transitions.

#### State Space and Transition Probabilities

For the kangaroo algorithm:
- **State space**: S = G (the group being analyzed)
- **Transition operator**: The function F defines a deterministic transition: state y transitions to state F(y) with probability 1
- **Stationarity**: The walk eventually enters a cycle, but is not aperiodic in the classical sense since it follows a deterministic path

#### Non-Reversible Random Walks

Unlike many theoretical random walk models, Pollard kangaroo walks are inherently non-reversible. The transition from state i to state j does not necessarily have the same probability as the reverse transition j to i. This non-reversibility has important implications for:

1. **Mixing time analysis**: The spectral properties differ significantly from reversible chains
2. **Asymptotic variance**: Non-reversible dynamics can actually reduce variance in certain configurations
3. **Conductance bounds**: The standard conductance-based mixing time bounds require modification

#### Connection to Pseudo-Spectral Gap

For non-reversible Markov chains, the pseudo-spectral gap γ_ps controls mixing time from above and below, analogous to how the standard spectral gap controls reversible chains. For Pollard walks with partition-based step functions:

γ_ps depends on the eigenvalues of the transition matrix, which are determined by how well the partition structure (S_1, S_2, ..., S_r) separates the group G.

## 3. Mixing Properties

### What Makes a Walk Mix Well

A "mixing well" in the context of Pollard's kangaroo algorithm means that the walk rapidly approaches a near-uniform distribution across the group before finding a collision. Key characteristics of well-mixing walks:

#### Spectral Gap Requirements

The transition matrix P of the walk must have a spectral gap γ such that:
- All eigenvalues except the principal eigenvalue λ_1 = 1 satisfy |λ_i| ≤ 1 - γ
- Larger spectral gaps imply faster convergence to near-uniform distribution
- For partition-based walks, the spectral gap depends on the quality of the partition

#### Aperiodicity and Irreducibility

Though Pollard walks are deterministic, they should be designed to behave as if they were aperiodic and irreducible:
- Every reachable state can be visited with roughly equal probability over long sequences
- The period (if it exists) should be 1 to avoid cyclic behavior patterns

#### Expansion Properties

A well-mixing partition structure G = S_1 ⊔ S_2 ⊔ ... ⊔ S_r ensures good expansion:
- Each set S_i has size approximately |G|/r
- The partition boundary is "thin" relative to the set sizes
- Transitions between sets occur with roughly uniform probability

### Collision Efficiency Under Mixed Conditions

According to research on collision bounds:

**Expected collision time** when mixing has achieved near-stationarity approximately follows the birthday paradox:
```
E[T_collision] ≈ √(π|G|/2)
```

However, since the kangaroo algorithm terminates in an O(√N) interval rather than the full group |G|, the actual expected collision time is:
```
E[T_collision] ≈ √(π N/2)
```

where N is the size of the search interval [a,b].

### Importance of Mixing Before Collision

A critical insight from theoretical analysis is that the mixing time of the random walk directly affects the collision detection probability:

- **If r is fixed** (independent of |G|): Lower bounds on mixing time are exponential in log|G|, meaning collision time will be far from the optimal birthday bound
- **If r ≥ c·log|G|** (logarithmic dependence): Mixing time becomes polynomial (typically O((log|G|)²) to O((log|G|)³)), enabling collision times close to √(π|G|/2)

This explains why Teske's recommendation of r ≈ 20 partitions is empirically effective—for practical group sizes, 20 partitions provides sufficient separation while maintaining computational efficiency.

## 4. Jump Size Selection

### Number of Jump Sizes and Distribution

The kangaroo algorithm uses distinguished points and jump sizes determined by a hash function. The jump size function F: G → S where S is the set of possible jump sizes should satisfy:

#### Uniform Distribution Requirement

For each possible jump size s ∈ S:
```
Pr[F(g) = s] ≈ 1/|S| for all g ∈ G
```

This ensures that:
1. The algorithm doesn't exhibit bias toward particular region traversals
2. The probability of visiting any group element approaches uniform
3. The statistical properties match theoretical random walk models

#### Recommended Number of Jump Sizes

Research indicates that approximately **20 jump size categories** (corresponding to r ≈ 20 partitions) provides near-optimal performance:

- **Fewer jumps (r = 3)**: Simple implementation, but walks behave worse than true random walks
- **Moderate jumps (r = 8-12)**: Reasonable performance with moderate complexity
- **Optimal range (r = 15-25)**: Approximately 20% improvement over r = 3, approaching theoretical random walk performance
- **Excessive jumps (r > 50)**: Diminishing returns and increased precomputation overhead

#### Jump Size Magnitude

Jump sizes should be selected from a range that covers the search interval effectively. For interval [a,b] with N = b - a:

- **Minimum jump**: Should be at least 1 to avoid stalling
- **Maximum jump**: Typically O(√N) to balance coverage and collision probability
- **Distribution**: Geometric or exponential distributions often work better than uniform jump magnitudes, as they provide scale-free exploration

#### Composition of Jump Mappings

Teske proposed several effective jump function compositions:

1. **Additive walks**: Add a fixed jump value based on the current point's partition
   ```
   y_{i+1} = y_i + c_j where j = hash(y_i) mod r
   ```

2. **Mixed walks**: Combine additive and multiplicative components
   ```
   y_{i+1} = a_j · y_i + c_j where j = hash(y_i) mod r
   ```

3. **Squaring walks** (for certain group structures): Leverage the group operation efficiently

### Variance Reduction Through Jump Design

The variance of step sizes directly affects algorithm performance. Define step size variance as:
```
Var[S] = E[S²] - (E[S])²
```

Larger variance means some steps are much larger than others, leading to:
- Uneven exploration of the search space
- Higher actual collision times
- Less predictable distinguished point frequencies

Well-designed jump tables minimize variance by ensuring step sizes are relatively uniform in magnitude while still providing adequate coverage.

## 5. Partition Sets

### Optimal Partitioning Strategies

A partition is a division of the group G into disjoint subsets: G = S_1 ⊔ S_2 ⊔ ... ⊔ S_r.

#### Characteristics of Optimal Partitions

**Size balance**: |S_i| ≈ |G|/r for all i (within O(1) elements)

**Independence from structure**: The partition should be defined by a function ℓ: G → {1,2,...,r} that doesn't correlate with the group's algebraic properties

**Computational efficiency**: The partition function should be computable in O(1) time using fast operations (bit extractions, hash functions)

#### Partition Function Design

For elliptic curve groups (secp256k1 is standard), effective partition functions use the x-coordinate of curve points:

```
ℓ(P) = (x-coordinate of P) mod r
```

Or using bit extraction:
```
ℓ(P) = bits[k:k+log₂(r)] of (x-coordinate of P)
```

#### Distinguished Points Through Partitioning

The partition structure is intimately connected to distinguished point detection. In practical implementations:

1. **Primary partition**: All r sets S_1,...,S_r are used for the step function
2. **Distinguished point property**: A secondary partition identifies "distinguished points" (e.g., points whose x-coordinate begins with k zero bits)
3. **Collision detection**: When two kangaroos land on the same distinguished point, a collision has been found

### Efficiency of Multiple Small Partitions vs. Single Partition

**Single partition (r = 1)**:
- Effectively random walk with single step size
- Poor mixing properties
- Expected collision time significantly larger than √N

**Few partitions (r = 3)**:
- Pollard's original approach
- Better than r = 1, but still inferior to random walk performance
- Empirically performs about 1.2x slower than optimal

**Optimal partitions (r ≈ 20)**:
- Near-random walk behavior
- Approximately 20% speedup over r = 3
- Mixing time: O((log|G|)²) to O((log|G|)³)
- Achieves collision time close to theoretical √N bound

**Excessive partitions (r > 100)**:
- Negligible additional improvement
- Increased overhead in:
  - Jump table storage
  - Hash computation per step
  - Cache misses from larger jump tables

## 6. Expected Benefits

### Variance Reduction

Improving the step function from basic (r = 3) to optimized (r ≈ 20) provides variance reduction through multiple mechanisms:

#### Direct Variance of Step Sizes

**Basic function (r = 3)**:
- Three discrete jump sizes with significant magnitude differences
- High variance in step sizes leads to high variance in trajectory

**Optimized function (r ≈ 20)**:
- Twenty jump sizes with more granular magnitude distribution
- Lower variance enables more predictable walk behavior
- Closer approximation to continuous random walk

#### Algorithmic Variance

Beyond step size variance, the overall algorithm variance is reduced because:

1. **Improved mixing**: Faster approach to uniform distribution means more reliable collision probability
2. **Reduced tail risk**: Fewer outlier runs that take much longer than average
3. **Better parallelization**: More uniform walk lengths improve load balancing across processors

Empirical studies show:
```
Var[T_optimized] / Var[T_basic] ≈ 0.8-0.85
```

### Runtime Improvements

**Serial execution speedup**:
- Optimized step function: ~1.2x faster (20% improvement)
- This corresponds to the square of the mixing quality improvement

**Expected run time comparison**:
- Basic approach: ~1.2·√N group operations (empirically observed)
- Optimized approach: ~1.0·√N group operations (approaching theoretical bound)

**Parallel execution improvements**:
- More uniform walk lengths reduce synchronization overhead
- Better variance reduction means fewer stragglers in the last work units
- Linear speedup extends to larger processor counts

### Distinguished Point Benefits

The partition-based approach enables:
1. **Memory efficiency**: Only storing distinguished points, not full walks
2. **Collision detection**: Automatic via table lookup when distinguished points match
3. **Improved variance**: Distinguished point probability adjusts naturally with interval size

Expected distinguished point frequency:
```
Pr[point is distinguished] = 1/2^(dpBit)
```

For a search interval of size N with dpBit chosen such that approximately √N points are stored:
```
dpBit ≈ log₂(2√N) ≈ 0.5·log₂(N)
```

## 7. Implementation Considerations

### Changes to the Step Function

Converting from basic to optimized kangaroo implementation requires:

#### 1. Expand Partition Count

**Before (Basic)**:
```
partition_count = 3
step_function = if bit[0] == 0: multiply by g₁
                elif bit[0] == 1: multiply by g₂
                else: multiply by g₃
```

**After (Optimized)**:
```
partition_count = 20  // or 16 for power-of-2
partition_index = hash(current_point) % partition_count
step_function = add c[partition_index] to current_point
```

#### 2. Implement Efficient Partition Function

For elliptic curves, use bit extraction from x-coordinate:

```c
// Compute partition index from point x-coordinate
uint32_t get_partition(point_t p, int num_partitions) {
    // Extract bits from x-coordinate
    // num_partitions should be 2^k for efficiency
    return (p.x >> bit_offset) & (num_partitions - 1);
}

// With num_partitions = 16:
uint32_t partition_idx = (p.x >> 32) & 0xF;
```

#### 3. Precompute Jump Constants

Store all constants used in partition-based jumps:

```c
// For additive walk: y_{i+1} = y_i + c[j]
point_t jump_constants[NUM_PARTITIONS];  // Precomputed

// For mixed walk: y_{i+1} = a[j] * y_i + c[j]
scalar_t mult_constants[NUM_PARTITIONS];
point_t add_constants[NUM_PARTITIONS];
```

#### 4. Distinguished Point Detection

Implement efficient distinguished point checking:

```c
// Check if point is distinguished
// Distinguished: x-coordinate has dp_bits leading zero bits
bool is_distinguished_point(point_t p, int dp_bits) {
    // Extract top bits of x-coordinate
    uint64_t top_bits = p.x >> (256 - dp_bits);
    return top_bits == 0;
}

// Alternative: check specific bit pattern
bool is_distinguished_point_alt(point_t p, int dp_bits) {
    // More flexible: check any bit pattern
    return (p.x & dp_mask) == dp_pattern;
}
```

### Changes to Jump Table

#### 1. Size Expansion

- **Old**: 3 precomputed jump constants (or multipliers/addition pairs)
- **New**: 16-20 precomputed jump constants

Memory impact:
- Basic: 3 × sizeof(point) ≈ 3 × 64 bytes = 192 bytes (for compressed points)
- Optimized: 20 × 64 bytes = 1,280 bytes (minimal impact)

#### 2. Initialization Strategy

```c
// Precompute jump table
void init_jump_table(point_t* jump_table, int count, generator_t g) {
    for (int i = 0; i < count; i++) {
        // Each jump constant is determined by hash or derivation
        // Option 1: Deterministic generation
        scalar_t scalar = hash_to_scalar(i);
        jump_table[i] = scalar_multiply(g, scalar);

        // Option 2: Random generation (must be fixed for reproducibility)
        jump_table[i] = random_point();  // With fixed seed
    }
}
```

#### 3. Cache Optimization

For maximum efficiency:
- Keep jump table in L1/L2 cache (1-2 MB typically available)
- Arrange data for sequential access patterns
- Consider SIMD loading for batch operations in parallel code

#### 4. Consistency Across Parallel Workers

Critical: All parallel workers must use the same jump table to ensure walks merge when they collide:

```c
// Global/shared jump table
__global__ point_t global_jump_table[MAX_PARTITIONS];

// All workers reference the same table
__device__ void kangaroo_step(point_t* current, int worker_id) {
    uint32_t partition = get_partition(*current, NUM_PARTITIONS);
    elliptic_add(current, &global_jump_table[partition]);
}
```

### Performance Tuning Parameters

| Parameter | Basic | Optimized | Notes |
|-----------|-------|-----------|-------|
| partition_count | 3 | 16-20 | Powers of 2 for fast mod |
| dp_bits | Varies | log₂(2√N) | Adjust for memory |
| jump_table_size | 3 | 16-20 | bytes = 64 × count |
| walk_length_avg | ~1.2√N | ~1.0√N | Expected group ops |
| variance_reduction | 1.0x | 0.8-0.85x | Relative variance |
| speedup | 1.0x | 1.2x | Serial execution |

## 8. References

### Primary Research Papers

1. **Near Optimal Bounds for Collision in Pollard Rho for Discrete Log**
   - arXiv:math/0611586
   - Key Contribution: Proves collision occurs in O(√|G| log|G| log log|G|) steps using mixing time analysis
   - URL: https://arxiv.org/abs/math/0611586

2. **Kangaroo Methods for Solving the Interval Discrete Logarithm Problem**
   - Authors: Alex Fowler and Steven Galbraith
   - arXiv:1501.07019
   - URL: https://arxiv.org/abs/1501.07019

3. **How Long Does It Take to Catch a Wild Kangaroo?**
   - Authors: R. Montenegro and P. Tetali
   - Key Contribution: First rigorous proof of Pollard's kangaroo expected runtime
   - URL: https://arxiv.org/pdf/0812.0789

4. **Speeding Up Pollard's Rho Method For Computing Discrete Logarithms**
   - Author: Edlyn Teske
   - Key Contribution: Proposes r ≈ 20 partition-based step functions with ~20% speedup
   - URL: https://www.researchgate.net/publication/2771815

5. **Collision Bounds for the Additive Pollard Rho Algorithm**
   - Authors: Joppe W. Bos, Alina Dudeanu, and Dimitar Jetchev
   - URL: https://eprint.iacr.org/2012/087

### Supplementary References

6. **Mathematical Aspects of Mixing Times in Markov Chains**
   - URL: https://tetali.math.gatech.edu/PUBLIS/survey.pdf

7. **Computing Discrete Logarithms in an Interval**
   - URL: https://eprint.iacr.org/2010/617.pdf

8. **Time-Memory Trade-offs for Parallel Collision Search Algorithms**
   - URL: https://eprint.iacr.org/2017/581.pdf

---

**Document Date**: January 2026
**Status**: Technical reference for step function optimization in Pollard's kangaroo algorithm
