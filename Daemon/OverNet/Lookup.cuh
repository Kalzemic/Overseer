#include "defines.h"
class LookupTable
{
private:
    __device__ int hash(unsigned long long addr);

    __device__ void remove(unsigned long long addr);

public:
    int *head;
    unsigned long long *buffer;
    unsigned long long *map_keys;
    int *map_values;

    LookupTable();
    __device__ int insert(unsigned long long addr);
    __device__ int lookup(unsigned long long addr);
    __device__ unsigned long long addr_lookup(int rank);
    ~LookupTable();
};