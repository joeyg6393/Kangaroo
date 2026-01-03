Here is a focused list of **concrete upgrades** to JeanLucPons’ Kangaroo, each with:

- What the upgrade is (at a high level)
- What it buys you
- The key paper to read for the *logic* behind it

***

## 1. Equivalence classes in the exponent space

**Upgrade:**
Introduce an **equivalence‑class mapping** on exponents so that multiple exponents in the interval map to a **single representative**. The kangaroo walk runs over representatives; when a solution is found, you invert the mapping to recover the original exponent.

**Benefit:**

- Shrinks the *effective* interval length by the average class size.
- Reduces expected running time roughly by the same factor, for modest overhead in pre/post‑processing.

**Main logic source:**

- *Using Equivalence Classes to Accelerate Solving the Discrete Logarithm Problem in a Short Interval* (Galbraith, Ruprai, PKC 2010)[^1]
- *Using Equivalent Class to Solve Interval Discrete Logarithm Problem* (Qi, Ma, Lv, ICICS 2019)[^2]

These describe how to define the equivalence relation, choose class representatives, and update the kangaroo steps so the walk stays in the quotient space.

***

## 2. Multi‑kangaroo patterns per herd (3, 4, 5, 7 kangaroos)

**Upgrade:**
Inside each GPU “herd”, replace the basic “1 tame + 1 wild” scheme with **Fowler–Galbraith multi‑kangaroo patterns** (3‑, 4‑, or 5‑kangaroo setups). The kernel maintains several logically distinct walks that are coordinated to minimize the constant factor in the $\sqrt{N}$ runtime.

**Benefit:**

- Reduces the constant in the expected time $c \sqrt{N}$ compared to the classic two‑kangaroo algorithm.
- This improvement stacks with parallelization across GPUs.

**Main logic source:**

- *Kangaroo Methods for Solving the Interval Discrete Logarithm Problem* (Fowler, Galbraith, 2015)[^3]

This paper gives the exact multi‑kangaroo configurations, probabilities, and constants, and shows which patterns are optimal in theory.

***

## 3. Improved collision/distingished‑point (DP) handling via GPU/CPU hash tables

**Upgrade:**
Replace or augment the current DP storage with a **modern high‑throughput hash table** design:

- GPU‑side: compact cuckoo / bucketed hash table for local DP buffering.
- CPU‑side: high‑load‑factor hash table tuned for batched inserts/lookups.

**Benefit:**

- Higher DP insertion/lookup throughput.
- Supports denser DP usage (smaller `dpBits`) without bottlenecking, improving time‑to‑collision given large host RAM.

**Main logic source:**

- *Better GPU Hash Tables* (Awad et al., arXiv:2108.07232)[^4]
- *Compact Parallel Hash Tables on the GPU* (Hegeman et al., arXiv:2406.09255)[^5]
- *Hashinator: a portable hybrid hashmap designed for heterogeneous systems* (2024)[^6]

These give the design and complexity of GPU/heterogeneous hash tables suitable for distinguished‑point workloads.

***

## 4. Walk / step‑function tuning using collision bounds

**Upgrade:**
Redesign the **step function and partitioning** (how you choose jump sizes based on the current point) to have better mixing and collision properties, informed by modern analysis of Pollard‑type walks.

**Benefit:**

- Reduces variance and slightly improves expected runtime by avoiding poorly mixing walks.
- Helps choose the number and distribution of jump sizes and partition sets.

**Main logic source:**

- *Factoring and Discrete Logarithms using Pseudorandom Walks* (Galbraith, crypto‑book chapter 14)[^7]
- *Near Optimal Bounds for Collision in Pollard Rho for Discrete Log* (Dinur et al., 2007)[^8]

These explain how to model the walk as a Markov chain, how mixing affects collision time, and how to choose good step functions.

***

## 5. Collision detection time–space trade‑offs

**Upgrade:**
Adopt a **time–space optimized collision detection scheme** (e.g., variants of distinguished points and checkpoints) rather than a single ad‑hoc DP density. Use available RAM (like 1 TB) explicitly in the parameter choice.

**Benefit:**

- Systematic way to choose DP density and table size for a given RAM budget and target runtime.
- Potentially reduces wasted work after a collision occurs but before it is noticed.

**Main logic source:**

- *An Efficient Collision Detection Method for Computing Discrete Logarithms* (Li, Wang, Xu, 2012)[^9]

This paper analyzes collision detection strategies and gives explicit time–space trade‑off formulas that can be mapped to kangaroo’s DP scheme.

***

If you want, the next step can be a code‑oriented sketch: for each upgrade, which part of JeanLucPons’ Kangaroo codebase to touch (random function, kernel state layout, DP buffer, host hash table, etc.) and what the minimal prototype would look like.

<div align="center">⁂</div>

[^1]: http://link.springer.com/10.1007/978-3-642-13013-7_22

[^2]: http://link.springer.com/10.1007/978-3-030-41579-2_23

[^3]: https://arxiv.org/abs/1501.07019

[^4]: https://arxiv.org/pdf/2108.07232.pdf

[^5]: https://arxiv.org/pdf/2406.09255.pdf

[^6]: https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1407365/full

[^7]: http://arxiv.org/pdf/2405.14381.pdf

[^8]: https://arxiv.org/pdf/math/0611586.pdf

[^9]: https://onlinelibrary.wiley.com/doi/10.1155/2012/635909

