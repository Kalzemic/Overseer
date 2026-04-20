#include "Lookup.cuh"
#include <iostream>
LookupTable::LookupTable()
{
    cudaError_t err;
    err = cudaMalloc(&head, sizeof(*head));
    if (err != cudaSuccess)
    {
        std::cerr << "failed to initialize header" << std::endl;
        std::terminate();
    }
    err = cudaMalloc(&buffer, sizeof(*buffer) * VOCAB_SIZE);
    if (err != cudaSuccess)
    {
        std::cerr << "failed to initialize buffer" << std::endl;
        std::terminate();
    }
    err = cudaMalloc(&map_keys, sizeof(unsigned long long) * MAP_SIZE);
    if (err != cudaSuccess)
    {
        std::cerr << "failed to initialize map keys" << std::endl;
        std::terminate();
    }
    err = cudaMalloc(&map_values, sizeof(int) * MAP_SIZE);
    if (err != cudaSuccess)
    {
        std::cerr << "failed to initialize map values" << std::endl;
        std::terminate();
    }

    cudaMemset(head, 0, sizeof(int));
    cudaMemset(buffer, -1, sizeof(unsigned long long) * VOCAB_SIZE);
    cudaMemset(map_keys, 0xFF, sizeof(unsigned long long) * MAP_SIZE);
}

__device__ int LookupTable::hash(unsigned long long addr)
{
    addr ^= addr >> 33;
    addr *= 0xff51afd7ed558ccdULL;
    addr ^= addr >> 33;
    addr *= 0xc4ceb9fe1a85ec53ULL;
    addr ^= addr >> 33;
    return (int)(addr & (MAP_SIZE - 1));
}
__device__ int LookupTable::insert(unsigned long long addr)
{
    int slot = hash(addr);
    while (map_keys[slot] != MAP_EMPTY && map_keys[slot] != MAP_DELETED)
    {
        slot = (slot + 1) & (MAP_SIZE - 1);
    }
    int rank = *head;
    auto old_addr = buffer[rank];
    remove(old_addr);
    buffer[rank] = addr;
    map_keys[slot] = addr;
    map_values[slot] = rank;
    *head = (rank + 1) & (VOCAB_SIZE - 1);
    return rank;
}

__device__ void LookupTable::remove(unsigned long long addr)
{
    if (addr == MAP_EMPTY || addr == MAP_DELETED)
        return;
    int slot = hash(addr);

    while (map_keys[slot] != MAP_EMPTY && map_keys[slot] != MAP_DELETED)
    {
        if (map_keys[slot] == addr)
        {
            map_keys[slot] = MAP_DELETED;
            int rank = map_values[slot];
            map_values[slot] = MAP_VALUE_DELETED;
            buffer[rank] = MAP_EMPTY;
            return;
        }
        slot = (slot + 1) & (MAP_SIZE - 1);
    }
}

__device__ int LookupTable::lookup(unsigned long long addr)
{
    int slot = hash(addr);

    while (map_keys[slot] != MAP_EMPTY && map_keys[slot] != MAP_DELETED)
    {
        if (map_keys[slot] == addr)
            return map_values[slot];

        slot = (slot + 1) & (MAP_SIZE - 1);
    }

    return -1;
}

__device__ unsigned long long LookupTable::addr_lookup(int rank)
{
    if (rank >= VOCAB_SIZE || rank < 0)
        return MAP_EMPTY;
    return buffer[rank];
}

LookupTable::~LookupTable()
{
    cudaFree(head);
    cudaFree(buffer);
    cudaFree(map_keys);
    cudaFree(map_values);
}
