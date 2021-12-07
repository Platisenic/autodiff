CXX = g++
CXXFLAGS = -Wall -Werror -Wextra -pedantic -std=c++11 -O2 -lgtest -lpthread -Iinclude
PROJECTFILES = $(wildcard include/autodiff/*.hpp)
TEST = tests/test_autodiff

.PHONY: all clean test

all: $(TEST)

$(TEST): $(TEST).cpp $(PROJECTFILES)
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -rf *.o $(TEST)

test: $(TEST)
	./$(TEST)
