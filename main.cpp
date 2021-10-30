#include "node.hpp"

#include <iostream>

int main(){
    Node a = Node("a", 5);
    Node b = Node("b", 6);
    Node c = (a+b)*b;
    std::cout << c.input1->getName() << std::endl;
    std::cout << c.input2->getName() << std::endl;
    std::cout << c.input1->input1->getName() << std::endl;
    std::cout << c.input1->input2->getName() << std::endl;

    return 0;
}
