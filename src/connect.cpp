

#include "connect.h"
#include <thread>
#include <mutex>
#include "secCompMultiParty.h"
#include <vector>
using namespace std;

#define STRING_BUFFER_SIZE 256
extern void error(string str);

// this player number
extern int partyNum;

// communication
string *addrs;
BmrNet **communicationSenders;
BmrNet **communicationReceivers;

// Communication measurements object
extern CommunicationObject commObject;

// setting up communication
void initCommunication(string addr, int port, int player, int mode)
{
	char temp[25];
	strcpy(temp, addr.c_str());
	if (mode == 0)
	{
		communicationSenders[player] = new BmrNet(temp, port);
		communicationSenders[player]->connectNow();
	}
	else
	{
		communicationReceivers[player] = new BmrNet(port);
		communicationReceivers[player]->listenNow();
	}
}

void initializeCommunication(int *ports)
{
	int i;
	communicationSenders = new BmrNet *[NUM_OF_PARTIES];
	communicationReceivers = new BmrNet *[NUM_OF_PARTIES];
	thread *threads = new thread[NUM_OF_PARTIES * 2];
	for (i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i != partyNum)
		{
			threads[i * 2 + 1] = thread(initCommunication, addrs[i], ports[i * 2 + 1], i, 0);
			threads[i * 2] = thread(initCommunication, "127.0.0.1", ports[i * 2], i, 1);
		}
	}
	for (int i = 0; i < 2 * NUM_OF_PARTIES; i++)
	{
		if (i != 2 * partyNum && i != (2 * partyNum + 1))
			threads[i].join(); // wait for all threads to finish
	}

	delete[] threads;
}

void initializeCommunicationSerial(int *ports) // Use this for many parties
{
	communicationSenders = new BmrNet *[NUM_OF_PARTIES];
	communicationReceivers = new BmrNet *[NUM_OF_PARTIES];
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i < partyNum)
		{
			initCommunication(addrs[i], ports[i * 2 + 1], i, 0);
			initCommunication("127.0.0.1", ports[i * 2], i, 1);
		}
		else if (i > partyNum)
		{
			initCommunication("127.0.0.1", ports[i * 2], i, 1);
			initCommunication(addrs[i], ports[i * 2 + 1], i, 0);
		}
	}
}

void initializeCommunication(char *filename, int p)
{
	FILE *f = fopen(filename, "r");
	partyNum = p;
	char buff[STRING_BUFFER_SIZE];
	char ip[STRING_BUFFER_SIZE];

	addrs = new string[NUM_OF_PARTIES];
	int *ports = new int[NUM_OF_PARTIES * 2];

	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		fgets(buff, STRING_BUFFER_SIZE, f);
		sscanf(buff, "%s\n", ip);
		addrs[i] = string(ip);
		// cout << addrs[i] << endl;
		ports[2 * i] = 31000 + i * NUM_OF_PARTIES + partyNum;
		ports[2 * i + 1] = 31000 + partyNum * NUM_OF_PARTIES + i;
	}

	fclose(f);
	initializeCommunicationSerial(ports);

	delete[] ports;
}

void bool2u8(vector<smallType> &res, const vector<bool> &data, size_t size)
{
	size_t i = 0; // point to data
	size_t j = 0; // point to res
	while (i < size)
	{
		smallType temp = 0;
		for (size_t k = 0; (k < 8) && (i < size); ++k)
		{
			temp = (temp << 1) + data[i];
			++i;
		}
		res[j] = temp;
		++j;
	}
}

void u82bool(vector<bool> &res, const vector<smallType> &data, size_t size)
{
	size_t i = 0; // point to data
	size_t j = 0; // point to res
	while (j < size && size - j > 7)
	{
		bitset<8> temp(data[i]);
		// smallType temp = data[i];
		for (int k = 7; k >= 0; --k)
		{
			res[j] = temp[k];
			++j;
		}
		++i;
	}

	size_t d = size - j;
	if (d > 0)
	{
		bitset<8> temp(data[i]);
		for (int k = d-1; k >= 0; --k)
		{
			res[j] = temp[k];
			++j;
		}
	}
}

void receiveBoolVector(vector<bool> &vec, size_t player, size_t size)
{
	vector<smallType> temp(size / 8 + 1);
	if (!communicationReceivers[player]->receiveMsg(temp.data(), temp.size() * sizeof(smallType), 0))
		cout << "Receive myType vector error" << endl;
	u82bool(vec, ref(temp), size);
}

void sendBoolVector(vector<bool> &vec, size_t player, size_t size)
{
	// cout << "send bool vec" << endl;
	vector<smallType> temp(size / 8 + 1);
	bool2u8(ref(temp), vec, size);

	if (!communicationSenders[player]->sendMsg(temp.data(), temp.size() * sizeof(smallType), 0))
		cout << "Receive myType vector error" << endl;
}

// synchronization functions
void sendByte(int player, char *toSend, int length, int conn)
{
	communicationSenders[player]->sendMsg(toSend, length, conn);
	// totalBytesSent += 1;
}

void receiveByte(int player, int length, int conn)
{
	char *sync = new char[length + 1];
	communicationReceivers[player]->receiveMsg(sync, length, conn);
	delete[] sync;
	// totalBytesReceived += 1;
}

void synchronize(int length)
{
	char *toSend = new char[length + 1];
	memset(toSend, '0', length + 1);
	vector<thread *> threads;
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i == partyNum)
			continue;
		for (int conn = 0; conn < NUMCONNECTIONS; conn++)
		{
			threads.push_back(new thread(sendByte, i, toSend, length, conn));
			threads.push_back(new thread(receiveByte, i, length, conn));
		}
	}
	for (vector<thread *>::iterator it = threads.begin(); it != threads.end(); it++)
	{
		(*it)->join();
		delete *it;
	}
	threads.clear();
	delete[] toSend;
}

void start_communication()
{
	if (commObject.getMeasurement())
		error("Nested communication measurements");

	commObject.reset();
	commObject.setMeasurement(true);
}

void pause_communication()
{
	if (!commObject.getMeasurement())
		error("Communication never started to pause");

	commObject.setMeasurement(false);
}

void resume_communication()
{
	if (commObject.getMeasurement())
		error("Communication is not paused");

	commObject.setMeasurement(true);
}

void end_communication(string str)
{
	cout << "----------------------------------------------" << endl;
	cout << "Communication, " << str << ", P" << partyNum << ": "
		 << (float)commObject.getSent() / 1000000 << "MB (sent) "
		 << (float)commObject.getRecv() / 1000000 << "MB (recv)" << endl;
	cout << "Rounds, " << str << ", P" << partyNum << ": "
		 << commObject.getRoundsSent() << "(sends) "
		 << commObject.getRoundsRecv() << "(recvs)" << endl;
	cout << "----------------------------------------------" << endl;
	commObject.reset();
}
