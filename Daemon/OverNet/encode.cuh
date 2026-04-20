
#include "Lookup.cuh"

void encode(LookupTable *page_table, LookupTable *ip_table, unsigned long long *pages, unsigned long long *ips, long long *tensor, cudaStream_t stream);

void decode(LookupTable *table, long long *ranks, unsigned long long *addrs, cudaStream_t stream);