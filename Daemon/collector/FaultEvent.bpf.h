#ifndef __FAULTEVENT_BPF__
#define __FAULTEVENT_BPF__

#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct page_fault_ctx
{
    unsigned short common_type;         // offset:0  size:2
    unsigned char common_flags;         // offset:2  size:1
    unsigned char common_preempt_count; // offset:3  size:1
    int common_pid;                     // offset:4  size:4
    unsigned long address;              // offset:8  size:8
    unsigned long ip;                   // offset:16 size:8
    unsigned long error_code;           // offset:24 size:8
};

#endif //__FAULTEVENT_BPF__