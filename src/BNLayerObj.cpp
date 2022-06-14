#pragma once
#include "BNLayerObj.h"
#include "Functionalities.h"
#include <cmath>
using namespace std;

BNLayerObj::BNLayerObj(BNConfig *conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->inputSize, conf->numBatches),
      //   gamma(conf->inputSize, make_pair(0, 0)),
      beta(conf->inputSize, make_pair(0, 0)),
      inv_sqrt(conf->inputSize),
      norm_x(conf->numBatches * conf->inputSize),
      beta_grad(conf->numBatches * conf->inputSize),
      gamma_grad(conf->numBatches * conf->inputSize),
      activations(conf->numBatches * conf->inputSize)
//   xhat(conf->numBatches * conf->inputSize),
//   sigma(conf->numBatches),
//   activations(conf->inputSize * conf->numBatches),
//   deltas(conf->inputSize * conf->numBatches)
{
    initialize();
};

void BNLayerObj::initialize()
{
    if (partyNum == PARTY_A) // gamma = 1
    {
        gamma = RSSVectorMyType(conf.inputSize, make_pair(1, 0));
    }
    else if (partyNum == PARTY_C)
    {
        gamma = RSSVectorMyType(conf.inputSize, make_pair(0, 1));
    }
    else
    {
        gamma = RSSVectorMyType(conf.inputSize, make_pair(0, 0));
    }
    size_t B = conf.numBatches;
    size_t m = conf.inputSize;
    size_t size = B * m;
};

void BNLayerObj::printLayer()
{
    cout << "----------------------------------------------" << endl;
    cout << "(" << layerNum + 1 << ") BN Layer\t\t  " << conf.inputSize << " x "
         << conf.numBatches << endl;
}

/**
 * @brief
 *
 * @param inputActivation
 */
void BNLayerObj::forward(const RSSVectorMyType &inputActivation)
{
    log_print("BN.forward");
    // size_t B = conf.numBatches;
    // size_t m = conf.inputSize;
    // size_t size = B * m;
    myType eps = (1e-5) * (1 << FLOAT_PRECISION);
    // size_t EPSILON = (myType)(1 << (FLOAT_PRECISION - 8));
    // TODO: Accept initialization from the paper
    // size_t INITIAL_GUESS = (myType)(1 << (FLOAT_PRECISION));
    // size_t SQRT_ROUNDS = 4;

    // vector<myType> eps(B, EPSILON), initG(B, INITIAL_GUESS);
    RSSVectorMyType var_eps(size, make_pair(0, 0)), mu(m, make_pair(0, 0)), b(B);
    // RSSVectorMyType divisor(B, make_pair(0, 0));

    // Compute mean
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            mu[j] = mu[j] + inputActivation[i * m + j];
    funcProbTruncation<RSSVectorMyType, myType>(mu, LOG_MINI_BATCH, m); //  1 truncation by batchSize [1, D]

    // Compute x - mean
    RSSVectorMyType x_mean(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            x_mean[i * m + j] = inputActivation[i * m + j] - mu[j];

    // Compute (x-mean)^2
    RSSVectorMyType temp2(size);
    funcDotProduct(x_mean, x_mean, temp2, size, true, FLOAT_PRECISION + LOG_MINI_BATCH);

    // mean((x-mean)^2)
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            var_eps[j] = var_eps[j] + temp2[i * m + j];
    // funcProbTruncation<RSSVectorMyType, myType>(var_eps, LOG_MINI_BATCH, m);

    // Compute (variance + epsilon)
    funcAddOneConst(var_eps, eps, m);

    // inver Square Root
    funcInverseSqrt<RSSVectorMyType, myType>(inv_sqrt, var_eps, m); // [1,D]
    for (int i = 0; i < m; ++i)                                     // scalling invsqrt
    {
        RSSMyType temp = var_eps[i];
        for (int j = 1; j < B; ++j)
            var_eps[j * m + i] = temp;
    }
    funcDotProduct<RSSVectorMyType, myType>(inv_sqrt, x_mean, norm_x, size, true, FLOAT_PRECISION); // [B, D] * [1, D]

    // self.gamma * self.norm_x + self.beta
    // Scaling
    RSSVectorMyType g_repeat(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            g_repeat[i * m + j] = gamma[j];
    funcDotProduct(gamma, norm_x, activations, size, true, FLOAT_PRECISION);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            activations[i * m + j] = activations[i * m + j] + beta[j];
}

void BNLayerObj::backward(const RSSVectorMyType &input_grad)
{
    //  self.beta_grad = np.sum(grad, axis=0)
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            beta_grad[j] = beta_grad[j] + input_grad[i * m + j];
    // funcProbTruncation<RSSVectorMyType, myType>(beta_grad, LOG_MINI_BATCH, m);

    // self.gamma_grad = np.sum(self.norm_x * grad, axis=0)    # 1 multiplication
    RSSVectorMyType temp(size);
    funcDotProduct(norm_x, input_grad, temp, size, true, FLOAT_PRECISION);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            gamma_grad[j] = gamma_grad[j] + temp[i * m + j];

    //  dxhat = grad * self.gamma   # 1 multiplication
    RSSVectorMyType dxhat(size);
    RSSVectorMyType g_repeat(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            g_repeat[i * m + j] = gamma[j];
    funcDotProduct(input_grad, g_repeat, dxhat, size, true, FLOAT_PRECISION);

    // self.act_grad = self.inv_sqrt * (B*dxhat - np.sum(dxhat, axis=0) - self.norm_x * np.sum(dxhat * self.norm_x, axis=0)) / B
    RSSVectorMyType bdxhat(size);
    funcMulConst(bdxhat, dxhat, B, size);            // B*dxhat
    RSSVectorMyType sumdxhat(size, make_pair(0, 0)); // np.sum(dxhat, axis=0)
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            sumdxhat[j] = sumdxhat[j] + dxhat[i * m + j];
    RSSVectorMyType dxx(size); // dxhat * self.norm_x
    funcDotProduct(dxhat, norm_x, dxx, size, true, FLOAT_PRECISION);
    RSSVectorMyType sumdxx(size); // np.sum(dxhat * self.norm_x, axis=0)
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            sumdxx[j] = sumdxx[j] + dxx[i * m + j];
    for (int i = 0; i < B; ++i)
        for (int j = 1; j < m; ++j)
            sumdxx[j] = sumdxx[j - m];
    funcDotProduct(norm_x, sumdxx, sumdxx, size, true, FLOAT_PRECISION);
    // (bdxhat-sumdxhat-sumdxx)
    for (size_t i = 0; i < size; i++)
    {
        bdxhat[i] = bdxhat[i] - sumdxhat[i] - sumdxx[i];
    }
    // self.inv_sqrt * ()/B
    RSSVectorMyType inv_repeat(size);
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < m; ++j)
            inv_repeat[i * m + j] = inv_sqrt[j];
    funcDotProduct(inv_repeat, bdxhat, activations, size, true, FLOAT_PRECISION + B);
}

// https://kevinzakka.github.io/2016/09/14/batch_normalization/
// void BNLayerObj::computeDelta(RSSVectorMyType &prevDelta)
// {
//     log_print("BN.computeDelta");

//     size_t B = conf.numBatches;
//     size_t m = conf.inputSize;

//     // Derivative with xhat
//     RSSVectorMyType g_repeat(B * m), dxhat(B * m);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             g_repeat[i * m + j] = gamma[i];

//     funcDotProduct(g_repeat, deltas, dxhat, B * m, true, FLOAT_PRECISION);

//     // First term
//     RSSVectorMyType temp1(B * m);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp1[i * m + j] = ((myType)m) * dxhat[i * m + j];

//     // Second term
//     RSSVectorMyType temp2(B * m, make_pair(0, 0));
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp2[i * m] = temp2[i * m] + dxhat[i * m + j];

//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp2[i * m + j] = temp2[i * m];

//     // Third term
//     RSSVectorMyType temp3(B * m, make_pair(0, 0));
//     funcDotProduct(dxhat, xhat, temp3, B * m, true, FLOAT_PRECISION);
//     for (int i = 0; i < B; ++i)
//         for (int j = 1; j < m; ++j)
//             temp3[i * m] = temp3[i * m] + temp3[i * m + j];

//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp3[i * m + j] = temp3[i * m];

//     funcDotProduct(temp3, xhat, temp3, B * m, true, FLOAT_PRECISION);

//     // Numerator
//     subtractVectors<RSSMyType>(temp1, temp2, temp1, B * m);
//     subtractVectors<RSSMyType>(temp1, temp3, temp1, B * m);

//     RSSVectorMyType temp4(B);
//     for (int i = 0; i < B; ++i)
//         temp4[i] = ((myType)m) * sigma[i];

//     funcBatchNorm(temp1, temp4, prevDelta, m, B);
// }

// void BNLayerObj::updateEquations(const RSSVectorMyType &prevActivations)
// {
//     log_print("BN.updateEquations");

//     size_t B = conf.numBatches;
//     size_t m = conf.inputSize;

//     // Update beta
//     RSSVectorMyType temp1(B, make_pair(0, 0));
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp1[i] = temp1[i] + deltas[i * m + j];

//     subtractVectors<RSSMyType>(beta, temp1, beta, B);

//     // Update gamma
//     RSSVectorMyType temp2(B * m, make_pair(0, 0)), temp3(B, make_pair(0, 0));
//     funcDotProduct(xhat, deltas, temp2, B * m, true, FLOAT_PRECISION);
//     for (int i = 0; i < B; ++i)
//         for (int j = 0; j < m; ++j)
//             temp3[i] = temp3[i] + temp2[i * m + j];

//     subtractVectors<RSSMyType>(gamma, temp3, gamma, B);
// }
