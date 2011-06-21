#include <stdlib.h>

float randFloat( float low, float upp ) {
    return low + ((float)rand()/(float)RAND_MAX)*(upp - low);
}
