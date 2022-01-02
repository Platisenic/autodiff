CXX = g++
CXXFLAGS = -Wall -Werror -Wextra -pedantic -std=c++14 -O2
PROJECTFILES = $(wildcard include/autodiff/*.hpp)
TEST = tests/test_autodiff
BIND = bind
BIND_SO_NAME = autodiff$(shell python3-config --extension-suffix)

.PHONY: all clean test lint

all: $(TEST) $(BIND_SO_NAME)

$(TEST): $(TEST).cpp $(PROJECTFILES)
	$(CXX) $< -o $@ $(CXXFLAGS) -lgtest -lpthread -Iinclude

$(BIND_SO_NAME): $(BIND).cpp $(PROJECTFILES)
	$(CXX) $< -o $(BIND_SO_NAME) $(CXXFLAGS) -shared -fPIC -Iinclude $(shell python3 -m pybind11 --includes)
	cp $(BIND_SO_NAME) tests/$(BIND_SO_NAME)

clean:
	rm -rf *.o $(TEST) $(BIND_SO_NAME) tests/$(BIND_SO_NAME) tests/__pycache__

test: all
	./$(TEST)
	pytest -vx

lint: 
	cpplint --filter=-legal/copyright --linelength=120 $(PROJECTFILES) 
