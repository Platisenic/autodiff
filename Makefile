CXX = g++
CXXFLAGS = -Wall -Werror -Wextra -pedantic -std=c++11 -O2 -lgtest -lpthread

TEST = test_autodiff
.PHONY: all clean test

all: $(TEST)

$(TEST): $(TEST).cpp autodiff.hpp
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -rf *.o $(TEST)

test: $(TEST)
	./$(TEST)
