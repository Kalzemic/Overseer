#pragma once
#include "defines.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstdint>

struct FaultReport
{
    int32_t pid;
    unsigned long long addrs[SEQ_LENGTH];
    unsigned long long ips[SEQ_LENGTH];
} __attribute__((packed));

struct EnforcerReport
{
    int pid;
    unsigned long long addrs[SEQ_LENGTH];
} __attribute__((packed));

struct UDPServer
{
    int sock;
    sockaddr_in addr;
    sockaddr_in clientAddr;
    socklen_t clientLen;

    bool init(uint16_t port);
    int receive(void *buffer, size_t size);
    void send(const void *buffer, size_t size);
    void close();
    void send(const void *buffer, size_t size, uint16_t port);
};