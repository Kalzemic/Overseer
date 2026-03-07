#include <bpf/libbpf.h>
#include "skel.bpf.h"
#include <iostream>
#include "FaultEvent.h"

#include "FaultCollector.hpp"
#include "Transport.hpp"

// const std::string devpath = "/dev/overseer";

int handle_fault(void *ctx, void *data, size_t data_sz)
{

    if (data_sz < sizeof(faultEvent))
    {
        return -1;
    }
    auto eve = *reinterpret_cast<faultEvent *>(data);
    auto hctx = reinterpret_cast<HandlerCtx *>(ctx);

    if (eve.cgroup_id != hctx->cgroup_id)
        return 0;

    auto &batch = hctx->inputManager[eve.pid];

    batch.addrs[batch.offset] = eve.addr;
    batch.ips[batch.offset] = eve.ip;
    batch.offset++;

    if (batch.offset >= BATCH_SIZE)
    {
        FaultReport report;
        report.pid = eve.pid;
        memcpy(report.addrs, batch.addrs, sizeof(batch.addrs));
        memcpy(report.ips, batch.ips, sizeof(batch.ips));
        int sent = hctx->transport->sendReport(report);

        if (sent < 0)
            perror("failed to send to jetson");

        batch.offset = 0;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "please provide cgroup id\n";
        return -1;
    }
    if (argc < 4)
    {
        std::cout << "must provide ip and port for overnet\n";
        return -1;
    }

    uint64_t cgroup_id;
    try
    {
        cgroup_id = std::stoull(argv[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Invalid cgroup id: " << argv[1] << " (" << e.what() << ")\n";
        return -1;
    }

    std::string overnet_ip(argv[2]);
    int overnet_port = std::stoi(argv[3]);

    SocketTransporter transporter(overnet_ip, overnet_port);

    HandlerCtx hctx{cgroup_id, &transporter, {}};

    auto skel = FaultCollect_bpf__open_and_load();
    if (!skel)
    {
        std::cerr << "Failed to open/load BPF skeleton\n";

        return -1;
    }

    if (FaultCollect_bpf__attach(skel) != 0)
    {
        std::cerr << "failed to attach to bpf program\n";

        FaultCollect_bpf__destroy(skel);
        return -1;
    }
    struct ring_buffer *rb = ring_buffer__new(
        bpf_map__fd(skel->maps.fault_buffer),
        handle_fault,
        &hctx,
        nullptr);
    if (!rb)
    {
        std::cerr << "failed to create ring buffer\n";

        FaultCollect_bpf__destroy(skel);
        return -1;
    }

    std::cout << "fault collection initialized\n";

    while (true)
    {
        int err = ring_buffer__poll(rb, 100 /*ms*/);
        if (err < 0)
        {
            std::cerr << "error polling ring buffer: " << strerror(-err) << "\n";
            break;
        }
    }

    ring_buffer__free(rb);
    FaultCollect_bpf__destroy(skel);

    return 0;
}