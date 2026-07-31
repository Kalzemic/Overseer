#ifndef _TRANSPORT_
#define _TRANSPORT_

#define INPUT_OFFSET 0
#include "defines.h"
#include <stdint.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
#include <stdexcept>

struct FaultReport
{
    int32_t pid;
    uint64_t addrs[BATCH_SIZE];
    uint64_t ips[BATCH_SIZE];
} __attribute__((packed));

class ITransporter
{
public:
    virtual int sendReport(const FaultReport &report) = 0;
    virtual ~ITransporter() = default;
};

class SocketTransporter : public ITransporter
{
    int sockfd{-1};
    struct sockaddr_in addr{};

public:
    SocketTransporter(std::string ip, int port);
    SocketTransporter(const SocketTransporter &other) = delete;
    SocketTransporter(SocketTransporter &&other) noexcept; 
    SocketTransporter &operator=(const SocketTransporter &other) = delete;
    SocketTransporter &operator=(SocketTransporter &&other) noexcept;
    ~SocketTransporter() override
    {
        if (sockfd >= 0)
            close(this->sockfd);
    }

    int sendReport(const FaultReport &report) override;
};

class PcieTransporter : public ITransporter
{
    int fd;
    void *ptr;
    size_t size;

public:
    PcieTransporter(std::string device, size_t size);
    PcieTransporter(const PcieTransporter &other) = delete;
    PcieTransporter operator=(const PcieTransporter &other) = delete;
    ~PcieTransporter();

    int sendReport(const FaultReport &report) override;
};

class ShmemTransporter : public ITransporter
{
    int fd;
    void *ptr;
    size_t size;

public:
    ShmemTransporter(std::string name, size_t size);
    ShmemTransporter(const ShmemTransporter &other) = delete;
    ShmemTransporter(ShmemTransporter &&other) noexcept;
    ShmemTransporter &operator=(const ShmemTransporter &other) = delete;
    ShmemTransporter &operator=(ShmemTransporter &&other) noexcept;
    ~ShmemTransporter();

    int sendReport(const FaultReport &report) override;
};

class TestTransporter : public ITransporter
{
public:
    TestTransporter() = default;
    ~TestTransporter() override = default;

    int sendReport(const FaultReport &report) override;
};

#endif // _TRANSPORT_
