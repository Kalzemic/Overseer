#ifndef _FAULT_EVENT_
#define _FAULT_EVENT_

struct faultEvent
{
    __u64 addr;
    __u64 ip;
    __u64 cgroup_id;
    __s32 pid;
    __s32 _pad;
};

#endif