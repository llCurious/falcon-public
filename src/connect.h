

#ifndef CONNECT_H
#define CONNECT_H

#include "basicSockets.h"
#include <sstream>
#include <vector>
#include "../util/TedKrovetzAesNiWrapperC.h"
#include <stdint.h>
#include <iomanip>
#include <fstream>
// #include <thread>
using namespace std;

extern BmrNet **communicationSenders;
extern BmrNet **communicationReceivers;

extern int partyNum;

// setting up communication
void initCommunication(string addr, int port, int player, int mode);
void initializeCommunication(int *ports);
void initializeCommunicationSerial(int *ports); // Use this for many parties
void initializeCommunication(char *filename, int p);

// bool type
void bool2u8(vector<smallType> &res, const vector<bool> &data, size_t size);
void u82bool(vector<bool> &res, const vector<smallType> &data, size_t size);
void receiveBoolVector(vector<bool> &vec, size_t player, size_t size);
void sendBoolVector(vector<bool> &vec, size_t player, size_t size);

// synchronization functions
void sendByte(int player, char *toSend, int length, int conn);
void receiveByte(int player, int length, int conn);
void synchronize(int length = 1);

void start_communication();
void pause_communication();
void resume_communication();
void end_communication(string str);

template <typename T>
void sendVector(const vector<T> &vec, size_t player, size_t size);
template <typename T>
void receiveVector(vector<T> &vec, size_t player, size_t size);

template <typename T>
void sendTwoVectors(const vector<T> &vec1, const vector<T> &vec2, size_t player, size_t size1, size_t size2);
template <typename T>
void receiveTwoVectors(vector<T> &vec1, vector<T> &vec2, size_t player, size_t size1, size_t size2);

template <typename T>
void sendThreeVectors(const vector<T> &vec1, const vector<T> &vec2, const vector<T> &vec3,
					  size_t player, size_t size1, size_t size2, size_t size3);
template <typename T>
void receiveThreeVectors(vector<T> &vec1, vector<T> &vec2, vector<T> &vec3,
						 size_t player, size_t size1, size_t size2, size_t size3);

template <typename T>
void sendFourVectors(const vector<T> &vec1, const vector<T> &vec2, const vector<T> &vec3,
					 const vector<T> &vec4, size_t player, size_t size1, size_t size2, size_t size3, size_t size4);
template <typename T>
void receiveFourVectors(vector<T> &vec1, vector<T> &vec2, vector<T> &vec3,
						vector<T> &vec4, size_t player, size_t size1, size_t size2, size_t size3, size_t size4);

template <typename T>
void sendSixVectors(const vector<T> &vec1, const vector<T> &vec2, const vector<T> &vec3,
					const vector<T> &vec4, const vector<T> &vec5, const vector<T> &vec6,
					size_t player, size_t size1, size_t size2, size_t size3, size_t size4, size_t size5, size_t size6);
template <typename T>
void receiveSixVectors(vector<T> &vec1, vector<T> &vec2, vector<T> &vec3,
					   vector<T> &vec4, vector<T> &vec5, vector<T> &vec6,
					   size_t player, size_t size1, size_t size2, size_t size3, size_t size4, size_t size5, size_t size6);

// template<typename T>
// void threadSend(const vector<T> &vec, size_t player, size_t size);
// template<typename T>
// void threadReceive(vector<T> &vec, size_t player, size_t size);

// template<typename T>
// void send1(const vector<T> &vec, size_t player, size_t size)
// {
// 	if(!communicationSenders[player]->sendMsg(vec.data(), (size) * sizeof(T), 1))
// 		cout << "Send vector error" << endl;
// }
// template<typename T>
// void send2(const vector<T> &vec, size_t player, size_t size)
// {
// 	if(!communicationSenders[player]->sendMsg(vec.data()+(size), (size) * sizeof(T), 0))
// 		cout << "Send vector error" << endl;
// }

// template<typename T>
// void recv1(vector<T> &vec, size_t player, size_t size)
// {
// 	if(!communicationReceivers[player]->receiveMsg(vec.data(), (size) * sizeof(T), 1))
// 		cout << "Receive vector error" << endl;
// }

// template<typename T>
// void recv2(vector<T> &vec, size_t player, size_t size)
// {
// 	if(!communicationReceivers[player]->receiveMsg(vec.data() + (size), (size) * sizeof(T), 0))
// 		cout << "Receive vector error" << endl;
// }

// template<typename T>
// void threadSend(const vector<T> &vec, size_t player, size_t size)
// {
// 	assert(sizeof(T) == 16 && "Hmm");
// 	assert(size%2 == 0 && "Send won't work");

// 	thread *threads = new thread[2];

// 	threads[0] = thread(send1<T>, ref(vec), player, size/2);
// 	threads[1] = thread(send2<T>, ref(vec), player, size/2);

// 	for (int i = 0; i < 2; i++)
// 		threads[i].join();

// 	delete[] threads;
// }

// template<typename T>
// void threadReceive(vector<T> &vec, size_t player, size_t size)
// {
// 	assert(sizeof(T) == 16 && "Hmm");
// 	assert(size%2 == 0 && "Send won't work");

// 	thread *threads = new thread[2];

// 	threads[0] = thread(recv1<T>, ref(vec), player, size/2);
// 	threads[1] = thread(recv2<T>, ref(vec), player, size/2);

// 	for (int i = 0; i < 2; i++)
// 		threads[i].join();

// 	delete[] threads;
// }

template <typename T>
void sendVector(const vector<T> &vec, size_t player, size_t size)
{
#if (LOG_DEBUG_NETWORK)
	cout << "Sending " << size * sizeof(T) << " Bytes to player " << player << " via ";
	if (sizeof(T) == 16)
		cout << "RSSMyType" << endl;
	else if (sizeof(T) == 8)
		cout << "myType" << endl;
	else if (sizeof(T) == 2)
		cout << "RSSSmallType" << endl;
	else if (sizeof(T) == 1)
		cout << "smallType" << endl;
#endif

	if (!communicationSenders[player]->sendMsg(vec.data(), size * sizeof(T), 0))
		cout << "Send vector error" << endl;
}

template <typename T>
void receiveVector(vector<T> &vec, size_t player, size_t size)
{
#if (LOG_DEBUG_NETWORK)
	cout << "Receiving " << size * sizeof(T) << " Bytes from player " << player << " via ";
	if (sizeof(T) == 16)
		cout << "RSSMyType" << endl;
	else if (sizeof(T) == 8)
		cout << "myType" << endl;
	else if (sizeof(T) == 2)
		cout << "RSSSmallType" << endl;
	else if (sizeof(T) == 1)
		cout << "smallType" << endl;
#endif

	if (!communicationReceivers[player]->receiveMsg(vec.data(), size * sizeof(T), 0))
		cout << "Receive myType vector error" << endl;
}

template <typename T>
void sendTwoVectors(const vector<T> &vec1, const vector<T> &vec2, size_t player, size_t size1, size_t size2)
{
	vector<T> temp(size1 + size2);
	for (size_t i = 0; i < size1; ++i)
		temp[i] = vec1[i];

	for (size_t i = 0; i < size2; ++i)
		temp[size1 + i] = vec2[i];

	sendVector<T>(temp, player, size1 + size2);
}

template <typename T>
void receiveTwoVectors(vector<T> &vec1, vector<T> &vec2, size_t player, size_t size1, size_t size2)
{
	vector<T> temp(size1 + size2);
	receiveVector<T>(temp, player, size1 + size2);

	for (size_t i = 0; i < size1; ++i)
		vec1[i] = temp[i];

	for (size_t i = 0; i < size2; ++i)
		vec2[i] = temp[size1 + i];
}

// Random size vectors allowed here.
template <typename T>
void sendThreeVectors(const vector<T> &vec1, const vector<T> &vec2, const vector<T> &vec3,
					  size_t player, size_t size1, size_t size2, size_t size3)
{
	vector<T> temp(size1 + size2 + size3);
	for (size_t i = 0; i < size1; ++i)
		temp[i] = vec1[i];

	for (size_t i = 0; i < size2; ++i)
		temp[size1 + i] = vec2[i];

	for (size_t i = 0; i < size3; ++i)
		temp[size1 + size2 + i] = vec3[i];

	sendVector<T>(temp, player, size1 + size2 + size3);
}

// Random size vectors allowed here.
template <typename T>
void receiveThreeVectors(vector<T> &vec1, vector<T> &vec2, vector<T> &vec3,
						 size_t player, size_t size1, size_t size2, size_t size3)
{
	vector<T> temp(size1 + size2 + size3);
	receiveVector<T>(temp, player, size1 + size2 + size3);

	for (size_t i = 0; i < size1; ++i)
		vec1[i] = temp[i];

	for (size_t i = 0; i < size2; ++i)
		vec2[i] = temp[size1 + i];

	for (size_t i = 0; i < size3; ++i)
		vec3[i] = temp[size1 + size2 + i];
}

template <typename T>
void sendFourVectors(const vector<T> &vec1, const vector<T> &vec2, const vector<T> &vec3,
					 const vector<T> &vec4, size_t player, size_t size1, size_t size2, size_t size3, size_t size4)
{
	vector<T> temp(size1 + size2 + size3 + size4);

	for (size_t i = 0; i < size1; ++i)
		temp[i] = vec1[i];

	for (size_t i = 0; i < size2; ++i)
		temp[size1 + i] = vec2[i];

	for (size_t i = 0; i < size3; ++i)
		temp[size1 + size2 + i] = vec3[i];

	for (size_t i = 0; i < size4; ++i)
		temp[size1 + size2 + size3 + i] = vec4[i];

	sendVector<T>(temp, player, size1 + size2 + size3 + size4);
}

template <typename T>
void receiveFourVectors(vector<T> &vec1, vector<T> &vec2, vector<T> &vec3,
						vector<T> &vec4, size_t player, size_t size1, size_t size2, size_t size3, size_t size4)
{
	vector<T> temp(size1 + size2 + size3 + size4);
	receiveVector<T>(temp, player, size1 + size2 + size3 + size4);

	for (size_t i = 0; i < size1; ++i)
		vec1[i] = temp[i];

	for (size_t i = 0; i < size2; ++i)
		vec2[i] = temp[size1 + i];

	for (size_t i = 0; i < size3; ++i)
		vec3[i] = temp[size1 + size2 + i];

	for (size_t i = 0; i < size4; ++i)
		vec4[i] = temp[size1 + size2 + size3 + i];
}

template <typename T>
void sendSixVectors(const vector<T> &vec1, const vector<T> &vec2, const vector<T> &vec3,
					const vector<T> &vec4, const vector<T> &vec5, const vector<T> &vec6,
					size_t player, size_t size1, size_t size2, size_t size3, size_t size4, size_t size5, size_t size6)
{
	vector<T> temp(size1 + size2 + size3 + size4 + size5 + size6);
	size_t offset = 0;

	for (size_t i = 0; i < size1; ++i)
		temp[i + offset] = vec1[i];

	offset += size1;
	for (size_t i = 0; i < size2; ++i)
		temp[i + offset] = vec2[i];

	offset += size2;
	for (size_t i = 0; i < size3; ++i)
		temp[i + offset] = vec3[i];

	offset += size3;
	for (size_t i = 0; i < size4; ++i)
		temp[i + offset] = vec4[i];

	offset += size4;
	for (size_t i = 0; i < size5; ++i)
		temp[i + offset] = vec5[i];

	offset += size5;
	for (size_t i = 0; i < size6; ++i)
		temp[i + offset] = vec6[i];

	sendVector<T>(temp, player, size1 + size2 + size3 + size4 + size5 + size6);
}

template <typename T>
void receiveSixVectors(vector<T> &vec1, vector<T> &vec2, vector<T> &vec3,
					   vector<T> &vec4, vector<T> &vec5, vector<T> &vec6,
					   size_t player, size_t size1, size_t size2, size_t size3, size_t size4, size_t size5, size_t size6)
{
	vector<T> temp(size1 + size2 + size3 + size4 + size5 + size6);
	size_t offset = 0;

	receiveVector<T>(temp, player, size1 + size2 + size3 + size4 + size5 + size6);

	for (size_t i = 0; i < size1; ++i)
		vec1[i] = temp[i + offset];

	offset += size1;
	for (size_t i = 0; i < size2; ++i)
		vec2[i] = temp[i + offset];

	offset += size2;
	for (size_t i = 0; i < size3; ++i)
		vec3[i] = temp[i + offset];

	offset += size3;
	for (size_t i = 0; i < size4; ++i)
		vec4[i] = temp[i + offset];

	offset += size4;
	for (size_t i = 0; i < size5; ++i)
		vec5[i] = temp[i + offset];

	offset += size5;
	for (size_t i = 0; i < size6; ++i)
		vec6[i] = temp[i + offset];
}

template <typename T>
void appendBool2u8(vector<T> &data, const vector<T> &adata, const vector<bool> &booldata, size_t size1, size_t size2)
{
	size_t i = 0; // point to data
	while (i < size1)
	{
		data[i] = adata[i];
		++i;
	}
	// size_t j = 0; // point to res
	int bit_num = sizeof(T) * 8;
	int j = 0;
	while (j < size2)
	{
		T temp = 0;
		for (size_t k = 0; (k < bit_num) && (j < size2); ++k)
		{
			temp = (temp << 1) + booldata[j];
			++j;
		}
		data[i] = temp;
		++i;
	}
}

/**
 * @brief split data to boold and adata respectively
 *
 * @tparam T
 * @param data
 * @param boold
 * @param adata
 * @param size1 size of boold
 * @param size2 size of adata
 */
template <typename T>
void splitu82Bool(const vector<T> &data, vector<T> &adata, vector<bool> &booldata, size_t size1, size_t size2)
{
	assert(booldata.size() == size1 && adata.size() == size2);
	size_t i = 0; // point to data
	while (i < size1)
	{
		adata[i] = data[i];
		++i;
	}

	size_t j = 0; // point to booldata
	int bit_num = sizeof(T) * 8;
	T temp;
	T msb = (1l << (bit_num - 1));

	bitset<64> msbbit(msb);
	// cout << "bit num " << bit_num << " msb " << msbbit << " ";
	while (j < size2 && size2 - j >= bit_num)
	{
		temp = data[i];
		cout << temp;
		for (int k = bit_num; k > 0; --k)
		{
			booldata[j] = (temp & msb) ? true : false;
			// cout << booldata[j] << "hh ";
			temp = temp << 1;
			// res[j] = temp[k];
			++j;
		}
		++i;
	}

	size_t d = size2 - j;
	if (d > 0)
	{
		msb = (msb >> (bit_num - d));
		temp = data[i];
		// cout << data[i] << " " << msb << "hh ";
		// bitset<bit_num> temp(data[i]);
		for (int k = d; k > 0; --k)
		{
			booldata[j] = (temp & msb) ? true : false;
			temp = temp << 1;
			++j;
		}
	}
}

#endif