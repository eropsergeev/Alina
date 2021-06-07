CXXFLAGS = -std=c++20 -Wall -Werror -pthread -march=native -D_GNU_SOURCE -fpic

CXX = g++-10

train: train.cpp fastrnn/*
	$(CXX) -o train train.cpp fastrnn/static.cpp alina_net.cpp $(CXXFLAGS) -Ofast

alina_net.so: alina_net.o
	$(CXX) -o alina_net.so alina_net.o fastrnn/static.cpp -shared -Ofast

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS) -Ofast

alina_net.o: alina_net.cpp alina_net.hpp fastrnn/tensor.hpp \
 fastrnn/executer.hpp fastrnn/barrier.hpp fastrnn/sysinfo.hpp \
 fastrnn/variable.hpp fastrnn/gru.hpp fastrnn/allocator.hpp \
 fastrnn/optimizer.hpp fastrnn/linear.hpp
train.o: train.cpp fft.hpp fastrnn/tensor.hpp fastrnn/executer.hpp \
 fastrnn/barrier.hpp fastrnn/sysinfo.hpp alina_net.hpp
alina_net.o: alina_net.hpp fastrnn/tensor.hpp fastrnn/executer.hpp \
 fastrnn/barrier.hpp fastrnn/sysinfo.hpp
fft.o: fft.hpp fastrnn/tensor.hpp fastrnn/executer.hpp \
 fastrnn/barrier.hpp fastrnn/sysinfo.hpp