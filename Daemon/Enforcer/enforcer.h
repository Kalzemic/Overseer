#ifndef __ENFORCER__
#define __ENFORCER__

#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kfifo.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include "defines.h"

struct prefetch_payload
{
    __s32 pid;
    __u64 addrs[BATCH_SIZE];
} __attribute__((packed));

extern wait_queue_head_t fifo_wait;
int server_start(int port);
void server_stop(void);
int server_run(void *data);

#endif //__ENFORCER__