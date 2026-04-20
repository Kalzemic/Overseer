#include "process_worker.h"

static DEFINE_HASHTABLE(workers, 8);

static int worker_run(void *data);

struct process_worker *find_worker(pid_t pid)
{
    struct process_worker *worker;
    hash_for_each_possible(workers, worker, node, pid)
    {
        if (worker->pid == pid)
        {
            return worker;
        }
    }
    return NULL;
}

struct process_worker *spawn_worker(pid_t pid)
{
    struct process_worker *worker;
    struct pid *pid_struct;
    struct task_struct *task;

    worker = kmalloc(sizeof(struct process_worker), GFP_KERNEL);
    if (!worker)
    {
        printk(KERN_ERR "overseer_enforcer: failed to allocate worker for pid %d\n", pid);
        return NULL;
    }

    pid_struct = find_get_pid(pid);
    if (!pid_struct)
    {
        printk(KERN_ERR "overseer_enforcer: failed to find pid %d\n", pid);
        kfree(worker);
        return NULL;
    }
    task = pid_task(pid_struct, PIDTYPE_PID);
    put_pid(pid_struct);
    if (!task)
    {
        printk(KERN_ERR "overseer_enforcer: failed to get task for pid %d\n", pid);
        kfree(worker);
        return NULL;
    }
    worker->mm = get_task_mm(task);
    if (!worker->mm)
    {
        printk(KERN_ERR "overseer_enforcer: failed to get mm for pid %d\n", pid);
        kfree(worker);
        return NULL;
    }
    worker->pid = pid;
    INIT_KFIFO(worker->addr_fifo);
    init_waitqueue_head(&worker->wait);

    worker->thread = kthread_run(worker_run, worker, "overseer_worker_%d", pid);
    if (IS_ERR(worker->thread))
    {
        printk(KERN_ERR "overseer_enforcer: failed to spawn kthread for pid %d\n", pid);
        mmput(worker->mm);
        kfree(worker);
        return NULL;
    }
    hash_add(workers, &worker->node, pid);
    printk(KERN_INFO "overseer_enforcer: spawned worker for pid %d\n", pid);
    return worker;
}

void destroy_worker(struct process_worker *worker)
{
    kthread_stop(worker->thread);
    mmput(worker->mm);
    hash_del(&worker->node);
    kfree(worker);
}
static int worker_run(void *data)
{
    struct process_worker *worker = (struct process_worker *)data;
    __u64 addrs[BATCH_SIZE];
    int i;

    while (!kthread_should_stop())
    {
        wait_event(worker->wait, kthread_should_stop() || !kfifo_is_empty(&worker->addr_fifo));

        if (kthread_should_stop())
            break;

        while (kfifo_out(&worker->addr_fifo, addrs, BATCH_SIZE) == BATCH_SIZE)
        {
            for (i = 0; i < BATCH_SIZE; i++)
            {
                struct page *page;
                int ret;
                unsigned long addr = addrs[i];

                if (addr == 0 || addr > USERSPACE_MAX)
                    continue;

                addr &= PAGE_MASK;

                mmap_read_lock(worker->mm);
                ret = get_user_pages_remote(worker->mm, addr, 1, FOLL_FORCE, &page, NULL);
                mmap_read_unlock(worker->mm);

                if (ret == 1)
                    put_page(page);
                else if (ret < 0)
                    pr_warn("overseer_enforcer: failed to prefetch 0x%lx for pid %d\n",
                            addr, worker->pid);
            }
        }
    }

    return 0;
}
void destroy_all_workers(void)
{
    struct process_worker *worker;
    struct hlist_node *tmp;
    int i;

    hash_for_each_safe(workers, i, tmp, worker, node)
        destroy_worker(worker);
}