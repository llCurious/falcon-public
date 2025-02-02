
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class BNConfig : public LayerConfig
{
public:
	size_t inputSize = 5;
	size_t numBatches = 4;

	BNConfig(size_t _inputSize, size_t _numBatches)
		: inputSize(_inputSize),
		  numBatches(_numBatches),
		  LayerConfig("BN"){};
};
