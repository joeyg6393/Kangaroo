/*
* This file is part of the BSGS distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CONSTANTSH
#define CONSTANTSH

#include <cstdint>

// Release number
inline constexpr const char* RELEASE = "2.3";

// Use symmetry (equivalence classes via negation map)
// Provides ~1.4x speedup by exploiting P ~ -P equivalence
#define USE_SYMMETRY
inline constexpr bool UseSymmetry = true;

// Number of random jumps (max 512 for GPU)
inline constexpr int NB_JUMP = 32;

// GPU group size
inline constexpr int GPU_GRP_SIZE = 128;

// GPU number of runs per kernel call
inline constexpr int NB_RUN = 64;

// Kangaroo types
inline constexpr uint32_t TAME = 0;
inline constexpr uint32_t WILD = 1;

// SendDP Period in seconds
inline constexpr double SEND_PERIOD = 2.0;

// Timeout before closing idle client connection in seconds
inline constexpr double CLIENT_TIMEOUT = 3600.0;

// Number of merge partitions
inline constexpr int MERGE_PART = 256;

#endif //CONSTANTSH
