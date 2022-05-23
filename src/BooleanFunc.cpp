#pragma once
#include "BooleanFunc.h"

void mergeBoolVec(RSSVectorBoolType &result, const vector<bool> &a1, const vector<bool> &a2, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        result[i] = make_pair(a1[i], a2[i]);
    }
}

// result = data.first ^ data.second
void mergeRSSVectorBool(vector<bool> &result, RSSVectorBoolType &data, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        result[i] = data[i].first ^ data[i].second;
    }
}

void funcBoolShareSender(RSSVectorBoolType &result, const vector<bool> &data, size_t size)
{

    PrecomputeObject->getZeroBShareSender(result, size);

    vector<bool> a1_xor_data(size);
    for (size_t i = 0; i < size; i++)
    {
        bool temp = data[i] ^ result[i].first;
        result[i].first = temp;
        a1_xor_data[i] = temp;
    }

    // send a1+x to prevparty
    thread sender(sendBoolVector, ref(a1_xor_data), prevParty(partyNum), size);
    sender.join();
}

void funcBoolShareReceiver(RSSVectorBoolType &result, int shareParty, size_t size)
{
    if (partyNum == prevParty(shareParty))
    {
        vector<bool> a3(size);
        PrecomputeObject->getZeroBSharePrev(a3, size);

        vector<bool> a1_xor_data(size);
        thread receiver(receiveBoolVector, ref(a1_xor_data), shareParty, size); // receive a1+x
        receiver.join();

        mergeBoolVec(result, a3, a1_xor_data, size);
    }
    else
    {
        PrecomputeObject->getZeroBShareReceiver(result, size);
    }
}

void funcBoolAnd(RSSVectorBoolType &result, const RSSVectorBoolType &a1, const RSSVectorBoolType &a2, size_t size)
{
    vector<bool> zero_rand(size);
    PrecomputeObject->getZeroBRand(zero_rand, size);

    vector<bool> result1(size);
    vector<bool> result2(size);

    for (size_t i = 0; i < size; i++)
    {
        result1[i] = (a1[i].first & a2[i].first) ^ (a1[i].first & a2[i].second) ^ (a1[i].second & a2[i].first) ^ zero_rand[i];
    }

    thread *threads = new thread[2];
    threads[0] = thread(sendBoolVector, ref(result1), prevParty(partyNum), size);
    threads[1] = thread(receiveBoolVector, ref(result2), nextParty(partyNum), size);

    for (int i = 0; i < 2; i++)
        threads[i].join();

    mergeBoolVec(result, result1, result2, size);
}