
// Lookup Table definitions
#define MAP_SIZE (1 << 17)
#define MAP_EMPTY 0xFFFFFFFFFFFFFFFFULL
#define MAP_DELETED 0xFFFFFFFFFFFFFFFEULL
#define MAP_VALUE_DELETED -1

// neural network parameters
#define INPUT_SIZE 1
#define SEQ_LENGTH 64
#define D_MODEL 256
#define D_STATE 256
#define RESET_MAX 32768
#define VOCAB_SIZE (1 << 13)

// Server config
#define PORT 9090
#define ENFORCER_PORT 9091

// reg defines
#define PAGE_NORM 4096