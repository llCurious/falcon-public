#ifndef UNITTESTS_H
#define UNITTESTS_H
#pragma once

#include <thread>
#include "Functionalities.h"
#include "BNConfig.h"
#include "BNLayer.h"
#include "BNLayerOpt.h"
#include <time.h>
#include "connect.h"
#include "basicSockets.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <chrono>

extern CommunicationObject commObject;

void benchWCExtension();
void benchMSExtension();
void benchBN();
template <typename Vec, typename T>
void benchBNAcc();
template <typename Vec, typename T>
void benchSoftMax();

/************Debug****************/

// one party has the plain value and secret share to other
void debugPartySS();    // void funcPartySS
void debugPairRandom(); // void Precompute::getPairRand(Vec &a, size_t size)
void debugZeroRandom(); // void Precompute::getZeroShareRand(vector<T> &a, size_t size)
void debugReduction();
void debugPosWrap();
void debugWCExtension();
void debugBoolAnd();
void debugMixedShareGen();
void debugMSExtension(); // funcMSExtension(RSSVectorHighType &output, RSSVectorLowType &input, size_t size)
template <typename Vec, typename T>
void debugProbTruncation();
void debugTruncAndReduce();
template <typename Vec, typename T, typename RealVec>
void debugRandBit();
template <typename Vec, typename T>
void debugReciprocal();
template <typename VEC, typename T>
void debugDivisionByNR();
template <typename Vec, typename T>
void debugInverseSqrt();
template <typename Vec, typename T>
void debugSoftmax();
template <typename Vec, typename T>
void benchTrunc();

void runTest(string str, string whichTest, string &network);

template <typename Vec, typename T, typename RealVec>
void debugRandBit()
{
    size_t size = 1000;

    RealVec b(size);
    vector<highBit> b_p(size);

    funcRandBit<Vec, T, RealVec>(b, size);

    // check
    funcReconstruct(b, b_p, size, "b plain", false);
    for (size_t i = 0; i < size; i++)
    {
        assert(b_p[i] == 0 || b_p[i] == 1);
    }
}

template <typename Vec, typename T>
void debugProbTruncation()
{
    int trunc_bits = 10;
    // high trunc
    size_t k = (sizeof(T) << 3) - 2;

    T temp = 1l << trunc_bits;

    size_t size = 5;
    vector<T> data(size);
    size_t i = 0;

    for (size_t i = 0; i < size; i++)
    {
        data[i] = (FLOAT_BIAS << (i + 1));
    }

    // data[i] = -(1l << (k));
    // i++;
    // for (; i < size / 4; i++)
    // {
    //     data[i] = data[i - 1] + temp;
    // }
    // data[i] = 0;
    // i++;
    // for (; i < size / 2; i++)
    // {
    //     data[i] = data[i - 1] - temp;
    // }
    // data[i] = 0;
    // i++;
    // for (; i < 3 * size / 4; i++)
    // {
    //     data[i] = data[i - 1] + temp;
    // }
    // data[i] = (1l << k) - 1;
    // i++;
    // for (; i < size / 4; i++)
    // {
    //     data[i] = data[i - 1] - temp;
    // }

    int checkParty = PARTY_B;
    Vec datahigh(size);
    funcPartySS<Vec, T>(datahigh, data, size, checkParty);
    // above: test data

    Vec datatrunc(size);
    funcProbTruncation<Vec, T>(datatrunc, datahigh, trunc_bits, size);
    funcProbTruncation<Vec, T>(datahigh, trunc_bits, size);

    // check
#if (!LOG_DEBUG)
    vector<T> trunc_plain(size);
    funcReconstruct<Vec, T>(datatrunc, trunc_plain, size, "trunc plain", false);
    vector<T> trunc_plain2(size);
    funcReconstruct<Vec, T>(datahigh, trunc_plain2, size, "trunc plain", false);
    if (checkParty == partyNum)
    {
        if (k == 62)
        {
            for (size_t i = 0; i < size; i++)
            {
                // cout << (long)trunc_plain[i] << " " << (((long)data[i]) >> trunc_bits) << endl;
                long temp = ((long)data[i]) >> trunc_bits;
                assert(((long)trunc_plain[i] == temp) || ((long)trunc_plain[i] == temp + 1) || ((long)trunc_plain[i] == temp - 1));
                assert(((long)trunc_plain2[i] == temp) || ((long)trunc_plain2[i] == temp + 1) || ((long)trunc_plain2[i] == temp - 1));
                // cout << bitset<64>(trunc_plain[i]) << " " << bitset<64>((data[i] >> trunc_bits)) << endl;
            }
        }
        else
        {
            for (size_t i = 0; i < size; i++)
            {
                int temp = ((int)data[i]) >> trunc_bits;
                assert(((int)trunc_plain[i] == temp) || ((int)trunc_plain[i] == temp + 1) || ((int)trunc_plain[i] == temp - 1));
                assert(((int)trunc_plain2[i] == temp) || ((int)trunc_plain2[i] == temp + 1) || ((int)trunc_plain2[i] == temp - 1));
                // cout << (long)trunc_plain[i] << " " << (((long)data[i]) >> trunc_bits) << endl;
                // cout << bitset<64>(trunc_plain[i]) << " " << bitset<64>((data[i] >> trunc_bits)) << endl;
            }
        }
    }
#endif
}

template <typename Vec, typename T>
void debugReciprocal()
{
    cout << "Debug Reciprocal" << endl;
    size_t size = 10;
    vector<T> data(size);
    for (size_t i = 0; i < size; i++)
    {
        data[i] = (FLOAT_BIAS << (i + 1));
    }
    printVectorReal<T>(data, "input", size);

    Vec input(size);
    int checkParty = PARTY_B;
    funcPartySS(input, data, size, checkParty);

    Vec output(size);
    funcReciprocal2(output, input, false, size);

    vector<T> result(size);
    funcReconstruct<Vec, T>(output, result, size, "out", false);
    printVectorReal<T>(result, "output", size);
}

template <typename VEC, typename T>
void debugDivisionByNR()
{
    cout << "Debug Division Using NR" << endl;
    size_t size = 10000;
    vector<T> data(size);
    vector<T> que(size);

    size_t float_precision = FLOAT_PRECISION;
    if (std::is_same<T, highBit>::value)
    {
        float_precision = HIGH_PRECISION;
    }
    else if (std::is_same<T, lowBit>::value)
    {
        float_precision = LOW_PRECISION;
    }
    else
    {
        cout << "Not supported type" << typeid(data).name() << endl;
    }

    for (size_t i = 0; i < size; i++)
    {
        data[i] = (1 << (float_precision + i + 2));
        que[i] = (1 << (float_precision + i + 1));
    }
    printVectorReal<T>(data, "input", size);
    printVectorReal<T>(que, "que", size);

    VEC input(size);
    VEC quess(size);
    int checkParty = PARTY_B;
    funcPartySS(input, data, size, checkParty);
    funcPartySS(quess, que, size, checkParty);

    VEC output(size);
    funcDivisionByNR(output, input, quess, size);

    vector<T> result(size);
    funcReconstruct<VEC, T>(output, result, size, "out", false);
    printVectorReal<T>(result, "output", size);
}

template <typename Vec, typename T>
void debugInverseSqrt()
{
    cout << "Debug Inverse Sqrt" << endl;
    size_t size = 5;
    vector<T> data(size);
    for (size_t i = 0; i < size; i++)
    {
        data[i] = (FLOAT_BIAS << (i + 1));
    }
    printVectorReal<T>(data, "input", size);

    Vec input(size);
    int checkParty = PARTY_B;
    funcPartySS(input, data, size, checkParty);

    Vec output(size);
    funcInverseSqrt<Vec, T>(output, input, size);

    vector<T> result(size);
    funcReconstruct<Vec, T>(output, result, size, "out", false);
    printVectorReal<T>(result, "output", size);
}

template <typename Vec, typename T>
void debugSoftmax()
{
    size_t rows = 3, cols = 10;
    size_t size = rows * cols;

    vector<float> data_raw = {
        // 0.923921, 0.219685, -0.414526, 0.34122, -0.166053, -0.629501, 0.270858, 0.0569448, -0.283571, 0.261101,
        // 0.342496, 0.491277, -0.405227, 0.104489, 0.0440607, -0.558957, 0.213509, 0.14997, -0.0229616, 0.318014,
        // 0.394563, 0.396146, -0.558138, 0.357716, -0.0295143, -0.772969, 0.377216, 0.0590944, -0.0873451, 0.325517,
        2.60365, -0.290006, -0.721366, 0.507656, -0.434107, -0.595984, 0.532808, -0.0694342, -0.377345, -0.0523729,
        -0.124932, 0.807279, -0.534957, -0.0588999, 0.0464973, -0.855241, 0.116036, 0.219089, 0.0108147, 0.756191,
        0.41907, 0.301184, -0.434992, 0.190973, 0.18747, -0.639831, 0.21955, 0.151983, -0.169142, 0.523242};

    vector<T> data(size);
    size_t float_precision = FLOAT_PRECISION;
    if (std::is_same<T, highBit>::value)
    {
        float_precision = HIGH_PRECISION;
    }
    else if (std::is_same<T, lowBit>::value)
    {
        float_precision = LOW_PRECISION;
    }
    else
    {
        cout << "Not supported type" << typeid(data).name() << endl;
    }

    for (size_t i = 0; i < size; i++)
        data[i] = data_raw[i] * (1 << float_precision);

    Vec a(size), b(size);

    funcGetShares(a, data);

    funcSoftmax(a, b, rows, cols, false);

    // #if (!LOG_DEBUG)
    vector<T> output(size);
    funcReconstruct(b, output, size, "output", false);
    printVectorReal(data, "input", size);
    printVectorReal(output, "output", size);
    // print_vector(a, "FLOAT", "a_data:", size);
    // print_vector(b, "FLOAT", "b_data:", size);
    // #endif
}

template <typename Vec, typename T>
void debugBNLayer();
template <typename Vec, typename T>
void debugBNLayer()
{
    cout << "Debug Batch Normalization Layer" << endl;
    size_t B = 4, D = 5;
    size_t size = B * D;

    // Floating point representation
    vector<float> x_raw = {
        1, 2, 3, 4, 5,
        1, 3, 5, 7, 8,
        1, 2, 3, 6, 6,
        1, 2, 4, 5, 6};

    vector<float> grad_raw = {
        1, 2, 3, 4, 5,
        1, 3, 5, 7, 8,
        1, 2, 3, 6, 6,
        1, 2, 4, 5, 6};

    // FXP representation
    vector<T> x_p(size), grad_p(size);
    size_t float_precision = FLOAT_PRECISION;
    if (std::is_same<T, highBit>::value)
    {
        float_precision = HIGH_PRECISION;
    }
    else if (std::is_same<T, lowBit>::value)
    {
        float_precision = LOW_PRECISION;
    }
    else
    {
        cout << "Not supported type" << typeid(x_p).name() << endl;
    }

    for (size_t i = 0; i < size; i++)
    {
        x_p[i] = x_raw[i] * (1 << float_precision);
        grad_p[i] = grad_raw[i] * (1 << (10 + float_precision));
    }

    // Public to secret
    Vec input_act(size), grad(size);
    funcGetShares(input_act, x_p);
    funcGetShares(grad, grad_p);

    BNConfig *bn_conf = new BNConfig(D, B);
    BNLayerOpt *layer = new BNLayerOpt(bn_conf, 0);
    layer->printLayer();

    // Forward.
    Vec forward_output(size), backward_output(size);
    layer->forward(input_act);
    forward_output = *layer->getActivation();
    // print_vector(forward_output, "FLOAT", "BN Forward", size);
    vector<T> xfp(size);
    funcReconstruct(forward_output, xfp, size, "xgrad", true);

    // Backward.
    Vec x_grad(size);
    // layer->backward(grad);
    *(layer->getDelta()) = grad;
    layer->computeDelta(x_grad);
    vector<T> xgradp(size);
    funcReconstruct(x_grad, xgradp, size, "xgrad", true);
    // print_vector(x_grad, "FLOAT", "BN Backward- X", size);

    // // Noted: i recommend print the calculated delta for beta and gamma in BNLayerOpt.
    // layer->updateEquations(input_act);
}

template <typename Vec>
void getSoftMaxInput(Vec &a, size_t size)
{
    size_t float_precision = FLOAT_PRECISION;
    if (std::is_same<Vec, RSSVectorHighType>::value)
    {
        float_precision = HIGH_PRECISION;
    }
    else if (std::is_same<Vec, RSSVectorLowType>::value)
    {
        float_precision = LOW_PRECISION;
    }
    else
    {
        cout << "Not supported type" << typeid(a).name() << endl;
    }

    vector<float> data_raw(size);
    for (int i = 0; i < size; i++)
    {
        data_raw[i] = rand() % 10000 / 12300.0;
    }

    vector<highBit> data(size);

    for (size_t i = 0; i < size; i++)
        data[i] = data_raw[i] * (1 << float_precision);

    funcGetShares(a, data);
}

template <typename Vec, typename T>
void getSoftMaxInput(string filename, Vec &a, size_t B, size_t D)
{
    size_t size = B * D;
    vector<T> x_raw(size);

    size_t float_precision = FLOAT_PRECISION;
    if (std::is_same<Vec, RSSVectorHighType>::value)
    {
        float_precision = HIGH_PRECISION;
    }
    else if (std::is_same<Vec, RSSVectorLowType>::value)
    {
        float_precision = LOW_PRECISION;
    }
    else
    {
        cout << "Not supported type" << typeid(x_raw).name() << endl;
    }
    cout << B << " " << D << " " << size << endl;

    ifstream infile;
    infile.open(filename.data());
    assert(infile.is_open());
    int i = 0, j = 0;
    char buf[1048576] = {0};
    cout << "read ";
    while (infile >> buf)
    {
        x_raw[i * D + j] = stof(buf) * (1 << float_precision);
        ;
        // cout << i * D + j << " " << x_raw[i * D + j] << " " << stof(buf) << endl;

        ++j;
        if (j == D)
        {
            j = 0;
            ++i;
        }
    }
    cout << endl;
    infile.close();

    // for (size_t i = 0; i < size; i++)
    // {
    //     x_raw[i] = x_raw[i] * (1 << float_precision);
    //     cout << x_raw[i] << " ";
    // }
    // cout << endl;

    funcGetShares(a, x_raw);
}

template <typename Vec, typename T>
void benchSoftMax()
{
    if (std::is_same<T, lowBit>::value && MP_FOR_DIVISION && MP_FOR_INV_SQRT)
    {
        cout << "mix" << endl;
    }
    else if (std::is_same<T, highBit>::value)
    {
        cout << "high" << endl;
    }
    else if (std::is_same<T, lowBit>::value && !MP_FOR_DIVISION && !MP_FOR_INV_SQRT)
    {
        cout << "low" << endl;
    }
    // d=10/200，batch=100/1000/10000/100000
    size_t ds[2] = {10, 200};
    size_t batchs[3] = {100, 1000, 10000};
    // size_t ds[1] = {200};
    // size_t batchs[1] = {10000};
    size_t cnt = 3;

    size_t size;

    uint64_t round = 0;
    uint64_t commsize = 0;

    for (int d : ds)
    {
        for (int batch : batchs)
        {
            size = d * batch;

            Vec a(size), b(size);
            // getSoftMaxInput(a, size);

            // comm
            round = commObject.getRoundsRecv();
            commsize = commObject.getRecv();

            funcSoftmax<Vec>(a, b, batch, d, false);

            cout << "round: " << commObject.getRoundsRecv() - round << "  size: " << commObject.getRecv() - commsize << endl;

            // time
            double time_sum = 0;
            for (size_t i = 0; i < cnt; i++)
            {
                getSoftMaxInput(a, size);
                auto start = std::chrono::system_clock::now();
                funcSoftmax<Vec>(a, b, batch, d, false);
                auto end = std::chrono::system_clock::now();
                double dur = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6);
                cout << dur << endl;
                time_sum += dur;
            }
            cout << batch << " " << d << " " << time_sum / cnt << endl;
        }
    }
}

template <typename Vec, typename T>
void benchSoftMaxAcc()
{
    size_t d = 10;
    size_t batch = 3;
    string inf = "./scripts/test_data/soft_input.csv";
    string outf = "./scripts/test_data/";
    if (std::is_same<T, lowBit>::value && MP_FOR_DIVISION && MP_FOR_INV_SQRT)
    {
        cout << "mix" << endl;
        outf += "soft_output_mix.csv";
    }
    else if (std::is_same<T, lowBit>::value)
    {
        if (!MP_FOR_DIVISION && !MP_FOR_INV_SQRT)
        {
            cout << "lowbit" << endl;
            outf += "soft_output_l.csv";
        }
        else
        {
            outf += "soft_output_temp.csv";
            cout << MP_FOR_DIVISION << " " << MP_FOR_INV_SQRT << endl;
            // cout << "global conf error" << endl;
            // return;
        }
    }
    else
    {
        cout << "highbit" << endl;
        outf += "soft_output_h.csv";
    }

    size_t size;

    uint64_t round = 0;
    uint64_t commsize = 0;

    clock_t start, end;

    size = d * batch;

    Vec a(size), b(size);

    getSoftMaxInput<Vec, T>(inf, a, batch, d);

    funcSoftmax<Vec>(a, b, batch, d, false);

    vector<T> result(size);
    funcReconstruct(b, result, size, "result", false);
    mat2file<T>(result, outf, batch, d);

    // record output
}

template <typename T>
void truncInput(vector<T> &data, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand();
    }
}

template <typename Vec, typename T>
void benchTrunc()
{
    if (IS_FALCON)
    {
        cout << "falcon ";
    }
    else
    {
        cout << "ours ";
    }
    int trunc_bits = 13;
    if (std::is_same<T, lowBit>::value)
    {
        cout << "trunc 32" << endl;
    }
    else if (std::is_same<T, highBit>::value)
    {
        trunc_bits = 20;
        cout << "trunc 64" << endl;
    }
    else
    {
        cout << "not support" << endl;
    }
    size_t dims[4] = {100, 1000, 10000, 100000};
    int cnt = 10;
    uint64_t round = 0;
    uint64_t commsize = 0;

    commObject.setMeasurement(true);
    for (int i = 0; i < 4; i++)
    {
        size_t size = dims[i];
        cout << "dim " << size << endl;

        vector<T> data(size);
        truncInput<T>(data, size);
        Vec input(size), output(size);
        funcGetShares<Vec, T>(input, data);

        round = commObject.getRoundsRecv();
        commsize = commObject.getRecv();
        if (IS_FALCON)
        {
            funcTruncatePublic(input, trunc_bits, size);
        }
        else
        {
            funcProbTruncation<Vec, T>(output, input, trunc_bits, size);
        }

        cout << "round " << commObject.getRoundsRecv() - round << endl;
        // cout << "send round " << commObject.getRoundsSent() << endl;
        cout << "size " << commObject.getRecv() - commsize << endl;
        // cout << "send size" << commObject.getSent() << endl;
    }
    commObject.setMeasurement(false);
    for (int i = 0; i < 4; i++)
    {
        size_t size = dims[i];
        cout << "dim " << size << endl;
        double time_sum = 0;
        for (int j = 0; j < cnt; ++j)
        {
            vector<T> data(size);
            truncInput<T>(data, size);
            Vec input(size), output(size);
            funcGetShares<Vec, T>(input, data);

            auto start = std::chrono::system_clock::now();
            if (IS_FALCON)
            {
                funcTruncatePublic(input, trunc_bits, size);
            }
            else
            {
                funcProbTruncation<Vec, T>(output, input, trunc_bits, size);
            }
            auto end = std::chrono::system_clock::now();
            double dur = (std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6);
            time_sum += dur;
            // cout << j << " " << dur << endl;
        }
        cout << time_sum / cnt << endl;
    }
}
#endif