#include <linux/net.h>
#include <linux/in.h>
#include <net/sock.h>
#include "enforcer.h"

static struct socket *sock = NULL;

int server_run(void *data)
{
    struct kfifo *payload_fifo = (struct kfifo *)data;
    struct prefetch_payload payload;
    struct kvec iov =
        {
            .iov_base = &payload,
            .iov_len = sizeof(payload)};

    struct msghdr msg = {};

    while (!kthread_should_stop())
    {
        int len = kernel_recvmsg(sock, &msg, &iov, 1, sizeof(payload), MSG_WAITALL);
        if (len < 0)
        {
            if (len == -EINTR)
                continue;
            printk(KERN_ERR "overseer_enforcer: recv error %d\n", len);
            break;
        }
        if (len < sizeof(payload))
            continue;
        if (kfifo_is_full(payload_fifo))
        {
            struct prefetch_payload dropped;
            unsigned int ret = kfifo_out(payload_fifo, &dropped, 1);
            (void)ret;
        }
        printk(KERN_INFO "overseer_enforcer: payload received \n");
        kfifo_in(payload_fifo, &payload, 1);
        wake_up(&fifo_wait);
    }
    return 0;
}

int server_start(int port)
{
    struct sockaddr_in addr =
        {
            .sin_family = AF_INET,
            .sin_port = htons(port),
            .sin_addr.s_addr = htonl(INADDR_ANY)};
    int ret;
    ret = sock_create_kern(&init_net, AF_INET, SOCK_DGRAM, IPPROTO_UDP, &sock);
    if (ret < 0)
    {
        printk(KERN_ERR "overseer_enforcer: failed to create socker\n");
        return ret;
    }
    ret = kernel_bind(sock, (struct sockaddr *)&addr, sizeof(addr));
    if (ret < 0)
    {
        printk(KERN_ERR "overseer_enforcer: failed to bind socket\n");
        return ret;
    }

    printk(KERN_INFO "overseer_enforcer: listening on port %d\n", port);
    return 0;
}

void server_stop(void)
{
    if (sock)
        sock_release(sock);
}