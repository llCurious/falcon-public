#include "CIFAR10Parser.h"
using namespace std;

int main(int argc, char** argv) {
    if (argc != 3)
		ERROR("Syntax: <program> <path-to-training-file> <path-to-testing-file>");

    assert(0 == parse(argv[1], TRAINING));
    assert(0 == parse(argv[2], TESTING));

    return 0;
}