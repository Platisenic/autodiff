CC = g++
CFLAGS = -Wall -std=c++11 -O2

.PHONY: clean all run

all: main

main.o: main.cpp node.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

main: main.o 
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -rf *.o main

run: main
	./main
