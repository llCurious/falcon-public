
#pragma once
#include "TedKrovetzAesNiWrapperC.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include "AESObject.h"

using namespace std;

AESObject::AESObject(std::string filename)
{
	ifstream f(filename);
	string str{istreambuf_iterator<char>(f), istreambuf_iterator<char>()};
	f.close();
	int len = str.length();
	char common_aes_key[len + 1];
	memset(common_aes_key, '\0', len + 1);
	strcpy(common_aes_key, str.c_str());
	AES_set_encrypt_key((unsigned char *)common_aes_key, 256, &aes_key);
}

__m128i AESObject::newRandomNumber()
{
	rCounter++;
	if (rCounter % RANDOM_COMPUTE == 0) // generate more random seeds
	{
		for (int i = 0; i < RANDOM_COMPUTE; i++)
			tempSecComp[i] = _mm_set1_epi32(rCounter + i); // not exactly counter mode - (rcounter+i,rcouter+i,rcounter+i,rcounter+i)
		AES_ecb_encrypt_chunk_in_out(tempSecComp, pseudoRandomString, RANDOM_COMPUTE, &aes_key);
	}
	return pseudoRandomString[rCounter % RANDOM_COMPUTE];
}

bool AESObject::getBoolRand()
{
	bool ret;

	if (boolCnt == 0)
	{
		__m128i rand128 = newRandomNumber();
		long long *temp = (long long *)&rand128;
		boolRandomNumber = temp[0];
	}

	ret = (boolRandomNumber & 1) ? true : false;
	boolRandomNumber = boolRandomNumber >> 1;

	++boolCnt;
	if (boolCnt == boolNum)
		boolCnt = 0;

	return ret;
}

myType AESObject::get64Bits()
{
	myType ret;

	if (random64BitCounter == 0)
		random64BitNumber = newRandomNumber();

	int x = random64BitCounter % 2;
	uint64_t *temp = (uint64_t *)&random64BitNumber;

	switch (x)
	{
	case 0:
		ret = (myType)temp[1];
		break;
	case 1:
		ret = (myType)temp[0];
		break;
	}

	random64BitCounter++;
	if (random64BitCounter == 2)
		random64BitCounter = 0;

	return ret;
}

smallType AESObject::get8Bits()
{
	smallType ret;

	if (random8BitCounter == 0)
		random8BitNumber = newRandomNumber();

	uint8_t *temp = (uint8_t *)&random8BitNumber;
	ret = (smallType)temp[random8BitCounter];

	random8BitCounter++;
	if (random8BitCounter == 16)
		random8BitCounter = 0;

	return ret;
}

lowBit AESObject::getLowBitRand()
{
	lowBit ret;

	if (random32BitCounter == 0)
		random32BitNumber = newRandomNumber();
	random32BitArray = (lowBit *)&random32BitNumber;

	ret = (lowBit)random32BitArray[random32BitCounter];

	random32BitCounter++;
	if (random32BitCounter == 4)
		random32BitCounter = 0;

	return ret;
}

highBit AESObject::getHighBitRand()
{
	highBit ret;

	if (random64BitCounter == 0)
		random64BitNumber = newRandomNumber();
	random64BitArray = (highBit *)&random64BitNumber;

	ret = (highBit)random64BitArray[random64BitCounter];

	random64BitCounter++;
	if (random64BitCounter == 2)
		random64BitCounter = 0;

	return ret;
}

smallType AESObject::randModPrime()
{
	smallType ret;

	do
	{
		ret = get8Bits();
	} while (ret >= BOUNDARY);

	return (ret % PRIME_NUMBER);
}

smallType AESObject::randNonZeroModPrime()
{
	smallType ret;
	do
	{
		ret = randModPrime();
	} while (ret == 0);

	return ret;
}

smallType AESObject::AES_random(int i)
{
	smallType ret;
	do
	{
		ret = get8Bits();
	} while (ret >= ((256 / i) * i));

	return (ret % i);
}
