
// Lookup Table definitions
#define MAP_SIZE (1 << 15)
#define MAP_EMPTY 0xFFFFFFFFFFFFFFFFULL
#define MAP_DELETED 0xFFFFFFFFFFFFFFFEULL
#define MAP_VALUE_DELETED -1

// neural network parameters
#define VOCAB_SIZE (1 << 14)
#define SEQ_LENGTH 64
#define D_MODEL 256
#define D_STATE 256

// Server config
#define PORT 9090
#define ENFORCER_PORT 9091