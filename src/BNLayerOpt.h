
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
    ForwardVecorType activations;
    // ForwardVecorType low_gamma;
    // ForwardVecorType low_beta;
    // ForwardVecorType inv_sqrt;
    // ForwardVecorType norm_x;


    BackwardVectorType high_activations;
    BackwardVectorType deltas;
    // RSSVectorMyType gamma;
    // RSSVectorMyType beta;
    // RSSVectorMyType xhat;
    // RSSVectorMyType sigma;
    BackwardVectorType gamma;
    BackwardVectorType beta;
    BackwardVectorType high_inv_sqrt;
    BackwardVectorType high_norm_x;
    BackwardVectorType xmu;
    BackwardVectorType var;
    BackwardVectorType beta_grad;
    BackwardVectorType gamma_grad;
    BackwardVectorType beta_velocity;
	BackwardVectorType gamma_velocity;
    
    size_t B;
    size_t channel;
    size_t width;
    size_t height;
    size_t size;

public:
    // Constructor and initializer
    BNLayerOpt(BNConfig *conf, int _layerNum);
    void initialize();

    // Functions
    void printLayer() override;
    void forward(const ForwardVecorType &input_act) override;
    void backward(const BackwardVectorType &input_grad);
    void computeDelta(BackwardVectorType &prevDelta) override;
    void updateEquations(const BackwardVectorType &prevActivations) override;

    // Mixed-precision funcs
	void weight_reduction() override;
	void activation_extension() override;
	void weight_extension() override;

    // Getters
    ForwardVecorType *getActivation() { return &activations; };
    BackwardVectorType* getHighActivation() {return &high_activations;};
    BackwardVectorType *getDelta() { return &deltas; };
    BackwardVectorType* getGamma() {return &gamma;};
	BackwardVectorType* getBeta() {return &beta;};
    BackwardVectorType* getGammaGrad() {return &gamma_grad;};
	BackwardVectorType* getBetaGrad() {return &beta_grad;};
};