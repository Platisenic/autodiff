#pragma once

#include <memory>
#include <vector>
#include <autodiff/node.hpp>

namespace autodiff {

struct Variable {
    std::shared_ptr<Node> VarNodePtr;

    Variable() : Variable(0.0) {}

    Variable(const Variable &o) : Variable(o.VarNodePtr) {}

    Variable(const std::shared_ptr<Node> &v) :
        VarNodePtr(std::make_shared<DepVarNode>(v))
        {}

    Variable(const double &v) :
        VarNodePtr(std::make_shared<IndVarNode>(v))
        {}

    Variable& operator=(const Variable& o) {
        if (this == &o) { return *this; }
        VarNodePtr = o.VarNodePtr;
        return *this;
    }

    Variable& operator=(const double &v) {
        *this = Variable(v);
        return *this;
    }

    Variable& operator=(const std::shared_ptr<Node> &v) {
        *this = Variable(v);
        return *this;
    }

    double values(){
        return VarNodePtr->value;
    }
};


}  // namespace autodiff
