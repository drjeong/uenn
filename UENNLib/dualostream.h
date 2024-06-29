/**
 * @file dualstream.h
 *
 * @brief This file is used to send message to command prompt and screen
 *
 *
 *
 */
#pragma once

#include <iostream>

class DualStreamBuf : public std::streambuf {
public:
	DualStreamBuf(std::ostream& stream1, std::ostream& stream2)
		: m_stream1(stream1), m_stream2(stream2) {}

	int overflow(int c) override {
		if (c != EOF) {
			if (m_stream1.rdbuf()->sputc(c) == EOF || m_stream2.rdbuf()->sputc(c) == EOF)
				return EOF;
		}
		return c;
	}

private:
	std::ostream& m_stream1;
	std::ostream& m_stream2;
};

class DualOstream : public std::ostream {
public:
	DualOstream(std::ostream& stream1, std::ostream& stream2)
		: std::ostream(&m_buf), m_buf(stream1, stream2) {}

private:
	DualStreamBuf m_buf;
};