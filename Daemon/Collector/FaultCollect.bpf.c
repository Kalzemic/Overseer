#include "vmlinux.h"
#include "FaultEvent.bpf.h"
#include "FaultEvent.h"

char LICENSE[] SEC("license") = "GPL";

struct
{
   __uint(type, BPF_MAP_TYPE_RINGBUF);
   __uint(max_entries, 256 * sizeof(struct faultEvent));
} fault_buffer SEC(".maps");

SEC("tracepoint/exceptions/page_fault_user")
int fault_callback(struct page_fault_ctx *ctx)
{
   struct faultEvent *eve;

   eve = bpf_ringbuf_reserve(&fault_buffer, sizeof(*eve), 0);
   if (!eve)
      return -1;

   eve->pid = bpf_get_current_pid_tgid() >> 32;
   eve->cgroup_id = bpf_get_current_cgroup_id();
   eve->addr = ctx->address;
   eve->ip = ctx->ip;

   bpf_ringbuf_submit(eve, 0);
   return -1;
}