#ifndef UNITTESTS_H
#define UNITTESTS_H
#pragma once

#include <thread>
#include "Functionalities.h"
#include "BNConfig.h"
#include "BNLayer.h"
#include "BNLayerObj.h"

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
    size_t size = 5;
    vector<T> data(size);
    vector<T> que(size);
    for (size_t i = 0; i < size; i++)
    {
        data[i] = (FLOAT_BIAS << (i + 2));
        que[i] = (FLOAT_BIAS << (i + 1));
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

void debugBNLayer();

#endif