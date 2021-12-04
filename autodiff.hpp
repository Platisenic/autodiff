#pragma once

#include <memory>


struct Node{
    double value;

    Node(const double &v): value(v) {}
    ~Node() {}
    virtual void prop(const double &output) = 0;
};

struct VarNode : Node{
    double grad;

    VarNode(const double &v): Node(v), grad(0.0) {}
};

struct IndVarNode: VarNode{
    IndVarNode(const double &v): VarNode(v) {}
    void prop(const double &output) override{
        grad += output;
    }
};

struct BinaryOpNode: Node{
    std::shared_ptr<Node> left, right;

    BinaryOpNode(const double &v,
        const std::shared_ptr<Node> l,
        const std::shared_ptr<Node> r) :
        Node(v),
        left(l),
        right(r)
        {}
};

struct AddOpNode: BinaryOpNode{
    AddOpNode(const double &v,
        const std::shared_ptr<Node> l,
        const std::shared_ptr<Node> r) :
        BinaryOpNode(v, l, r)
        {}

    void prop(const double &output) override{
        left->prop(output);
        right->prop(output);
    }
};

struct SubOpNode: BinaryOpNode{
    SubOpNode(const double &v,
        const std::shared_ptr<Node> l,
        const std::shared_ptr<Node> r) :
        BinaryOpNode(v, l, r)
        {}

    void prop(const double &output) override{
        left->prop(output);
        right->prop(-output);
    }
};

struct MulOpNode: BinaryOpNode{
    MulOpNode(const double &v,
        const std::shared_ptr<Node> l,
        const std::shared_ptr<Node> r) :
        BinaryOpNode(v, l, r)
        {}

    void prop(const double &output) override{
        left->prop(right->value * output);
        right->prop(left->value * output);
    }
};

struct DivOpNode: BinaryOpNode{
    DivOpNode(const double &v,
        const std::shared_ptr<Node> l,
        const std::shared_ptr<Node> r) :
        BinaryOpNode(v, l, r)
        {}

    void prop(const double &output) override{
        const auto recRight = 1.0 / right->value;
        left->prop(recRight * output);
        right->prop(recRight * recRight * (-left->value) * output);
    }
};

