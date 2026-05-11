#include "enforcer.h"
#include "process_worker.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Kalzemic");
MODULE_DESCRIPTION("Overseer Enforcer");
MODULE_VERSION("0.1");

static int port = 9091;

module_param(port, int, 0444);

MODULE_PARM_DESC(port, "UDP port to listen for prefetch hints");
static DECLARE_KFIFO(payload_fifo, struct prefetch_payload, FIFO_SIZE);
static struct task_struct *server_thread = NULL;
static struct task_struct *enforcer_thread = NULL;
DECLARE_WAIT_QUEUE_HEAD(fifo_wait);

static int enforcer_run(void *data)
{

    while (!kthread_should_stop())
    {
        wait_event(fifo_wait, kthread_should_stop() || !kfifo_is_empty(&payload_fifo));

        if (kthread_should_stop())
            break;

        struct prefetch_payload payload;
        while (kfifo_out(&payload_fifo, &payload, 1))
        {
            struct process_worker *worker = find_worker(payload.pid);
            if (!worker)
            {
                worker = spawn_worker(payload.pid);
            }
            if (!worker)
            {
                printk(KERN_ERR "overseer_enforcer: failed to get worker for pid %d\n", payload.pid);
                continue;
            }

            if (kfifo_is_full(&worker->addr_fifo))
            {
                __u64 dropped[BATCH_SIZE];
                if (kfifo_out(&worker->addr_fifo, dropped, BATCH_SIZE) == 0)
                    printk(KERN_WARNING "overseer_enforcer: failed to drop batch\n");
            }
            kfifo_in(&worker->addr_fifo, payload.addrs, BATCH_SIZE);
            wake_up(&worker->wait);
        }
    }
    return 0;
}

static int __init enforcer_init(void)
{
    int ret;

    INIT_KFIFO(payload_fifo);
    ret = server_start(port);
    if (ret < 0)
    {
        printk(KERN_ERR "overseer_enforcer: failed to start server\n");
        return ret;
    }

    server_thread = kthread_run(server_run, &payload_fifo, "overseer_server");
    if (IS_ERR(server_thread))
    {
        server_stop();
        return PTR_ERR(server_thread);
    }

    enforcer_thread = kthread_run(enforcer_run, NULL, "overseer_enforcer");
    if (IS_ERR(enforcer_thread))
    {
        kthread_stop(server_thread);
        server_stop();
        return PTR_ERR(enforcer_thread);
    }

    printk(KERN_INFO "overseer_enforcer: loaded\n");
    return 0;
}

static void __exit enforcer_exit(void)
{
    kthread_stop(enforcer_thread);
    kthread_stop(server_thread);
    destroy_all_workers();
    server_stop();
    printk(KERN_INFO "overseer_enforcer: unloaded\n");
}

module_init(enforcer_init);
module_exit(enforcer_exit);