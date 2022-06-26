
#pragma once
#include "MaxpoolConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class MaxpoolLayer : public Layer
{
private:
	MaxpoolConfig conf;
	ForwardVecorType activations;
	BackwardVectorType high_activations;
	BackwardVectorType deltas;
	RSSVectorSmallType maxPrime;

public:
	//Constructor and initializer
	MaxpoolLayer(MaxpoolConfig* conf, int _layerNum);

	//Functions
	void printLayer() override;
	void forward(const ForwardVecorType& inputActivation) override;
	void computeDelta(BackwardVectorType& prevDelta) override;
	void updateEquations(const BackwardVectorType& prevActivations) override;

	// Mixed-precision funcs
	void weight_reduction() override;
	void activation_extension() override;
	void weight_extension() override;

	//Getters
	ForwardVecorType* getActivation() {return &activations;};
	BackwardVectorType* getDelta() {return &deltas;};
};