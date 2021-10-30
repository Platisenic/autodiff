#pragma once
#include <string>

class Node{
public:
    Node* input1;
    Node* input2;
    Node* output;
    std::string name;
    int value;
    int grad;
public:
    Node() = default;
    Node(std::string n, int v):
        input1(nullptr),
        input2(nullptr),
        output(nullptr),
        name(n),
        value(v),
        grad(0)
        {}
    Node(Node *i1, Node *i2, std::string n, int v):
        input1(i1),
        input2(i2),
        output(nullptr),
        name(n),
        value(v),
        grad(0)
        {}
    std::string getName(){ return name; }
    int getValue(){ return value; }
    Node& operator+(Node &rhs){
        Node *out = new Node(this, &rhs, this->name + "+" + rhs.name, value + rhs.value);
        output = out;
        return *out;
    }
    // Node* operator-(Node &rhs){
    //     Node *out = new Node(this, &rhs, this->name + "-" + rhs.name, value - rhs.value);
    //     output = out;
    //     return out;
    // }
    Node& operator*(Node &rhs){
        Node *out = new Node(this, &rhs, this->name + "*" + rhs.name, value * rhs.value);
        output = out;
        return *out;
    }
    // Node* operator/(Node &rhs){
    //     Node *out = new Node(this, &rhs, this->name + "/" + rhs.name, value / rhs.value);
    //     output = out;
    //     return out;
    // }
};
