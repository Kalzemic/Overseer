#include "Transport.hpp"
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

//------------Jetson Methods--------------

SocketTransporter::SocketTransporter(std::string ip, int port)
{
    this->sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (this->sockfd < 0)
        throw std::runtime_error("failed to open socket");

    this->addr.sin_family = AF_INET;
    this->addr.sin_port = htons(port);

    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0)
    {
        std::cerr << "failed to establish connection with jetson\n";
        close(sockfd);
        throw std::runtime_error("failed connecting to jetson");
    }
}

int SocketTransporter::sendReport(const FaultReport &report)
{
    int sent = sendto(this->sockfd,
                      &report,
                      sizeof(report), 0,
                      reinterpret_cast<struct sockaddr *>(&this->addr),
                      sizeof(this->addr));
    return sent;
}

//---------------PCIE Methods ----------------------

PcieTransporter::PcieTransporter(std::string device, size_t size) : fd(-1), ptr(nullptr), size(sizeof(FaultReport))
{
    this->fd = open(device.c_str(), O_RDWR);
    if (!this->fd)
        throw std::runtime_error("failed to create device memory");

    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, INPUT_OFFSET);
    if (ptr == MAP_FAILED)
    {
        close(fd);
        throw std::runtime_error("failed to connect to device bar");
    }
}

PcieTransporter::~PcieTransporter()
{
    if (ptr && ptr != MAP_FAILED)
        munmap(ptr, size);

    if (this->fd >= 0)
        close(this->fd);
}

int PcieTransporter::sendReport(const FaultReport &report)
{
    if (!ptr || ptr == MAP_FAILED)
        return -1;

    memcpy(ptr, &report, sizeof(report));

    return sizeof(report);
}

//---------------Shmem Methods ----------------------

ShmemTransporter::ShmemTransporter(std::string name, size_t size) : size(size)
{
    this->fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (this->fd == -1)
        throw std::runtime_error("failed to open shared memory");

    if (ftruncate(fd, size) == -1)
        throw std::runtime_error("failed to allocated shared space");

    this->ptr = mmap(nullptr, this->size, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);
    if (this->ptr == MAP_FAILED)
    {
        close(this->fd);
        throw std::runtime_error("failed to mmap");
    }
}

ShmemTransporter::~ShmemTransporter()
{
    if (this->ptr && this->ptr != MAP_FAILED)
        munmap(this->ptr, this->size);
    if (this->fd != -1)
        close(this->fd);
}

ShmemTransporter::ShmemTransporter(ShmemTransporter &&other) noexcept
    : fd(other.fd), ptr(other.ptr), size(other.size)
{
    other.fd = -1;
    other.ptr = nullptr;
    other.size = 0;
}

ShmemTransporter &ShmemTransporter::operator=(ShmemTransporter &&other) noexcept
{
    if (this != &other)
    {
        if (this->ptr && this->ptr != MAP_FAILED)
        {
            munmap(this->ptr, this->size);
        }
        if (this->fd != -1)
        {
            close(this->fd);
        }

        this->fd = other.fd;
        this->ptr = other.ptr;
        this->size = other.size;

        other.fd = -1;
        other.ptr = nullptr;
        other.size = 0;
    }
    return *this;
}

int ShmemTransporter::sendReport(const FaultReport &report)
{
    if (!this->ptr)
        return -1;

    memcpy(this->ptr, &report, sizeof(report));
    return sizeof(report);
}

//---------------Shmem Methods ----------------------

int TestTransporter::sendReport(const FaultReport &report)
{
    std::cout << "=== FaultReport ===\n";
    std::cout << "PID: " << report.pid << "\n";
    std::cout << "Addresses:\n";
    for (size_t i = 0; i < BATCH_SIZE; ++i)
    {
        if (report.addrs[i] == 0)
            break; // stop if array not fully filled
        std::cout << "Page: 0x" << std::hex << report.addrs[i] << std::dec << "   IP: 0x" << std::hex << report.ips[i] << std::dec << "\n";
    }
    std::cout << "===================\n";
    return sizeof(report);
}