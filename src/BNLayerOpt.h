
#pragma once
#include "BNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

class BNLayerOpt : public Layer
{
private:
    BNConfig conf;
    // RSSVectorMyType activations;
    RSSVectorMyType deltas;
    // RSSVectorMyType gamma;
    // RSSVectorMyType beta;
    // RSSVectorMyType xhat;
    // RSSVectorMyType sigma;
    RSSVectorMyType gamma;
    RSSVectorMyType beta;
    RSSVectorMyType inv_sqrt;
    RSSVectorMyType inv_sqrt_rep;
    RSSVectorMyType norm_x;
    RSSVectorMyType beta_grad;
    RSSVectorMyType gamma_grad;
    RSSVectorMyType activations;
    size_t B;
    size_t m;
    size_t size;

public:
    // Constructor and initializer
    BNLayerOpt(BNConfig *conf, int _layerNum);
    void initialize();

    // Functions
    void printLayer() override;
    void forward(const RSSVectorMyType &input_act) override;
    void backward(const RSSVectorMyType &input_grad);
    void computeDelta(RSSVectorMyType &prevDelta) override;
    void updateEquations(const RSSVectorMyType &prevActivations) override;

    // Getters
    RSSVectorMyType *getActivation() { return &activations; };
    RSSVectorMyType *getDelta() { return &deltas; };
    RSSVectorMyType* getGamma() {return &gamma;};
	RSSVectorMyType* getBeta() {return &beta;};
};