#ifndef _FAULTCOLLECTOR_
#define _FAULTCOLLECTOR_
#include <unordered_map>
#include "defines.h"

#include "Transport.hpp"
struct Batch
{
    unsigned long addrs[BATCH_SIZE];
    unsigned long ips[BATCH_SIZE];
    int offset;
};

struct HandlerCtx
{
    uint64_t cgroup_id;
    ITransporter *transport;
    std::unordered_map<int32_t, Batch> inputManager;
};

#endif //_FAULTCOLLECTOR_