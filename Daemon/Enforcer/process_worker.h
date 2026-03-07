#ifndef __PROCESS_WORKER__
#define __PROCESS_WORKER__

#include <linux/kthread.h>
#include <linux/hashtable.h>
#include <linux/sched/mm.h>
#include <linux/mm.h>
#include <linux/pid.h>
#include <linux/sched.h>
#include <linux/kfifo.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include "defines.h"
struct process_worker
{
    pid_t pid;
    struct task_struct *thread;
    struct mm_struct *mm;
    DECLARE_KFIFO(addr_fifo, __u64, FIFO_SIZE *BATCH_SIZE);
    wait_queue_head_t wait;
    struct hlist_node node;
};

struct process_worker *find_worker(pid_t pid);
struct process_worker *spawn_worker(pid_t pid);
void destroy_worker(struct process_worker *worker);
void destroy_all_workers(void);

#endif //__PROCESS_WORKER__