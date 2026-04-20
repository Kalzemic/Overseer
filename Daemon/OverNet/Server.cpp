#include "Server.hpp"
#include <cstring>
#include <unistd.h>
#include <iostream>
bool UDPServer::init(uint16_t port)
{
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0)
    {
        std::cerr << "Failed to create socket" << std::endl;
        return false;
    }

    std::memset(&addr, 0, sizeof(addr));

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (sockaddr *)&addr, sizeof(addr)) < 0)
    {
        std::cerr << "failed to bind socket" << std::endl;
        return false;
    }

    clientLen = sizeof(clientAddr);
    std::cout << "Listening on port " << port << std::endl;
    return true;
}

int UDPServer::receive(void *buffer, size_t size)
{
    return recvfrom(sock, buffer, size, 0, (sockaddr *)&clientAddr, &clientLen);
}

void UDPServer::close()
{
    ::close(sock);
}
void UDPServer::send(const void *buffer, size_t size, uint16_t port)
{
    sockaddr_in dest = clientAddr;
    dest.sin_port = htons(port);
    sendto(sock, buffer, size, 0, (sockaddr *)&dest, sizeof(dest));
}