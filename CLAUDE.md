# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kangaroo is a high-performance Pollard's kangaroo ECDLP (Elliptic Curve Discrete Logarithm Problem) solver for SECP256K1 (Bitcoin's elliptic curve). It finds private keys from public keys within a known 125-bit interval using the distinguished point method with CPU and GPU (CUDA) acceleration.

## Build Commands

### Windows (Visual Studio 2019 + CUDA 10.2)
Open `VC_CUDA102/Kangaroo.sln`, select Release configuration, and build.

Older Visual Studio versions:
- VS 2015 + CUDA 8: `VC_CUDA8/`
- VS 2017 + CUDA 10: `VC_CUDA10/`

### Linux
```bash
# CPU only
make all

# With GPU (ccap = compute capability, e.g., 35, 52, 60)
make gpu=1 ccap=52 all

# Debug build
make gpu=1 ccap=52 debug=1 all
```

Requires: g++ 7.3+, CUDA SDK. Edit Makefile to set `CUDA` and `CXXCUDA` paths.

## Running

```bash
# Basic usage
./kangaroo config.txt

# With GPU
./kangaroo -gpu -gpuId 0,1 config.txt

# Server mode
./kangaroo -s -d 12 -w save.work -wi 300 -o result.txt config.txt

# Client mode
./kangaroo -t 0 -gpu -c server_ip -w kang.work -wi 600 config.txt

# Validate GPU kernel
./kangaroo -check config.txt
```

Config file format (all hex values):
```
[Start_Range]
[End_Range]
[PublicKey_1]
...
```

## Architecture

```
main.cpp              Entry point, CLI parsing
Kangaroo.h/cpp        Core algorithm orchestrator (2500+ lines)
                      - SolveKeyCPU()/SolveKeyGPU() - kangaroo hopping
                      - Run()/RunServer() - main loops

SECPK1/               Elliptic curve cryptography library
├── Int.h/cpp         256/512-bit big integer arithmetic
├── IntMod.cpp        Modular arithmetic (Pornin's delayed right shift inversion)
├── IntGroup.h/cpp    Batch modular inversion (Cheon's method)
├── Point.h/cpp       EC point operations (projective coordinates)
└── SECP256K1.h/cpp   SECP256K1 curve operations, generator tables

GPU/                  NVIDIA CUDA acceleration
├── GPUEngine.h/cpp   Device management, kernel launching
├── GPUEngine.cu      CUDA kernels (inline PTX assembly)
├── GPUMath.h         PTX macros (UADD, UMULLO, UMULHI, etc.)
└── GPUCompute.h      Main computation kernel

HashTable.h/cpp       Distinguished point storage (2^18 entries)
Network.cpp           TCP client/server protocol
Backup.cpp            Work file persistence
Thread.cpp            Cross-platform threading (pthread/Windows)
Merge.cpp             Work file merging
PartMerge.cpp         Partitioned work file merging
```

## Key Constants

In `Constants.h`:
- `NB_JUMP 32` - Number of random jumps (max 512 for GPU)
- `GPU_GRP_SIZE 128` - Threads per GPU group
- `NB_RUN 64` - Iterations per kernel call

In `HashTable.h`:
- `HASH_SIZE_BIT 18` - Hash table size (2^18 = 262144 entries)

## Algorithm

Pollard's kangaroo (lambda method) for interval ECDLP:
1. Create tame herd starting from range start, wild herd from center
2. Perform pseudorandom walks using jump table based on x-coordinate
3. Store distinguished points (x has dpBit leading zeros) in hash table
4. Detect collision between tame/wild kangaroos
5. Solve: `private_key = k1 + tame_distance - wild_distance`

Expected operations: 2.08 * sqrt(range_size)

## Work Files

- `-w file -wi N` saves progress every N seconds
- `-ws` includes kangaroo state (recommended for resumption)
- `-wm file1 file2 dest` merges work files
- `-winfo file` shows work file details
