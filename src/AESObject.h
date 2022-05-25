
#ifndef AESOBJECT_H
#define AESOBJECT_H

#pragma once
#include <algorithm>
#include <iostream>
#include "TedKrovetzAesNiWrapperC.h"
#include "globals.h"
using namespace std;

class AESObject
{
private:
	// AES variables
	__m128i pseudoRandomString[RANDOM_COMPUTE];
	__m128i tempSecComp[RANDOM_COMPUTE];
	unsigned long rCounter = -1;
	AES_KEY_TED aes_key;

	// Extraction variables
	__m128i random8BitNumber{0};
	uint8_t random8BitCounter = 0;
	__m128i random64BitNumber{0};
	uint8_t random64BitCounter = 0;
	__m128i random32BitNumber{0};
	uint8_t random32BitCounter = 0;

	lowBit *random32BitArray;
	highBit *random64BitArray;

	__m128i randomTNumber{0};
	int randomTCounter = 0;
	int randomTNum = 0;

	// Private extraction functions
	__m128i newRandomNumber();

	// Private helper functions
	smallType AES_random(int i);

	// bool random
	long long boolRandomNumber;
	int boolNum = 128;
	int boolCnt = 0;

public:
	// Constructor
	AESObject(std::string filename);

	// Randomness functions
	myType get64Bits();
	smallType get8Bits();

	lowBit getLowBitRand();
	highBit getHighBitRand();

	bool getBoolRand();

	template <typename T>
	T getRand();

	template <typename T>
	void getRand(vector<T> &a);

	template <typename Vec>
	void getRandPair(Vec &a);

	// Other randomness functions
	smallType randModPrime();
	smallType randNonZeroModPrime();
};

template <typename T>
T AESObject::getRand()
{
	T ret;

	if (randomTCounter == 0)
		randomTNum = 16 / sizeof(T);
	randomTNumber = newRandomNumber();

	T *temp = (T *)&randomTNumber;
	ret = (T)temp[randomTCounter];

	randomTCounter++;
	if (randomTCounter == randomTNum)
		randomTCounter = 0;

	return ret;
}

template <typename T>
void AESObject::getRand(vector<T> &a)
{
	size_t a_size = a.size();
	size_t type_num = (16 / sizeof(T));
	int j = 0;
	__m128i randomBitNumber{0};

	while (a_size > type_num)
	{
		a_size -= type_num;

		randomBitNumber = newRandomNumber();
		T *temp = (T *)&randomBitNumber;

		for (size_t i = 0; i < type_num; i++)
		{
			a[j] = (T)temp[i];
			j++;
		}
	}

	randomBitNumber = newRandomNumber();
	T *temp = (T *)&randomBitNumber;
	for (size_t i = 0; i < a_size; i++)
	{
		a[j] = (T)temp[i];
		j++;
	}
}

#endif