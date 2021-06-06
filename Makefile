CXXFLAGS = -std=c++20 -Wall -Werror -pthread -march=native -D_GNU_SOURCE

CXX = g++-10

trinan: train.cpp fastrnn/*
	$(CXX) -o train train.cpp fastrnn/static.cpp alina_net.cpp $(CXXFLAGS) -Ofast