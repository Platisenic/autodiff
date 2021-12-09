#pragma once

#include <memory>
#include <autodiff/node.hpp>
#include <autodiff/variable.hpp>

namespace autodiff {

std::shared_ptr<Node> operator+(const std::shared_ptr<Node> &l, const std::shared_ptr<Node> &r) {
    return std::make_shared<AddOpNode>(l->value + r->value, l, r);
}

std::shared_ptr<Node> operator+(const std::shared_ptr<Node> &l, const double &r) {
    return std::make_shared<AddOpNode>(l->value + r, l, std::make_shared<ConstantNode>(r));
}

std::shared_ptr<Node> operator+(const double &l, const std::shared_ptr<Node> &r) {
    return std::make_shared<AddOpNode>(l + r->value, std::make_shared<ConstantNode>(l), r);
}

std::shared_ptr<Node> operator+(const std::shared_ptr<Node> &l) {
    return l;
}

std::shared_ptr<Node> operator+(const Variable &l) {
    return l.VarNodePtr;
}

std::shared_ptr<Node> operator+(const Variable &l, const Variable &r) {
    return l.VarNodePtr + r.VarNodePtr;
}

std::shared_ptr<Node> operator+(const Variable &l, const std::shared_ptr<Node> &r) {
    return l.VarNodePtr + r;
}

std::shared_ptr<Node> operator+(const std::shared_ptr<Node> &l, const Variable &r) {
    return l + r.VarNodePtr;
}

std::shared_ptr<Node> operator+(const double &l, const Variable &r) {
    return l + r.VarNodePtr;
}

std::shared_ptr<Node> operator+(const Variable &l, const double &r) {
    return l.VarNodePtr + r;
}

std::shared_ptr<Node> operator-(const std::shared_ptr<Node> &l, const std::shared_ptr<Node> &r) {
    return std::make_shared<SubOpNode>(l->value - r->value, l, r);
}

std::shared_ptr<Node> operator-(const std::shared_ptr<Node> &l, const double &r) {
    return std::make_shared<SubOpNode>(l->value - r, l, std::make_shared<ConstantNode>(r));
}

std::shared_ptr<Node> operator-(const double &l, const std::shared_ptr<Node> &r) {
    return std::make_shared<SubOpNode>(l - r->value, std::make_shared<ConstantNode>(l), r);
}

std::shared_ptr<Node> operator-(const std::shared_ptr<Node> &l) {
    return std::make_shared<NegOpNode>(-l->value, l);
}

std::shared_ptr<Node> operator-(const Variable &l) {
    return -l.VarNodePtr;
}

std::shared_ptr<Node> operator-(const Variable &l, const Variable &r) {
    return l.VarNodePtr - r.VarNodePtr;
}

std::shared_ptr<Node> operator-(const Variable &l, const std::shared_ptr<Node> &r) {
    return l.VarNodePtr - r;
}

std::shared_ptr<Node> operator-(const std::shared_ptr<Node> &l, const Variable &r) {
    return l - r.VarNodePtr;
}

std::shared_ptr<Node> operator-(const double &l, const Variable &r) {
    return l - r.VarNodePtr;
}

std::shared_ptr<Node> operator-(const Variable &l, const double &r) {
    return l.VarNodePtr - r;
}

std::shared_ptr<Node> operator*(const std::shared_ptr<Node> &l, const std::shared_ptr<Node> &r) {
    return std::make_shared<MulOpNode>(l->value * r->value, l, r);
}

std::shared_ptr<Node> operator*(const std::shared_ptr<Node> &l, const double &r) {
    return std::make_shared<MulOpNode>(l->value * r, l, std::make_shared<ConstantNode>(r));
}

std::shared_ptr<Node> operator*(const double &l, const std::shared_ptr<Node> &r) {
    return std::make_shared<MulOpNode>(l * r->value, std::make_shared<ConstantNode>(l), r);
}

std::shared_ptr<Node> operator*(const Variable &l, const Variable &r) {
    return l.VarNodePtr * r.VarNodePtr;
}

std::shared_ptr<Node> operator*(const Variable &l, const std::shared_ptr<Node> &r) {
    return l.VarNodePtr * r;
}

std::shared_ptr<Node> operator*(const std::shared_ptr<Node> &l, const Variable &r) {
    return l * r.VarNodePtr;
}

std::shared_ptr<Node> operator*(const double &l, const Variable &r) {
    return l * r.VarNodePtr;
}

std::shared_ptr<Node> operator*(const Variable &l, const double &r) {
    return l.VarNodePtr * r;
}

std::shared_ptr<Node> operator/(const std::shared_ptr<Node> &l, const std::shared_ptr<Node> &r) {
    return std::make_shared<DivOpNode>(l->value / r->value, l, r);
}

std::shared_ptr<Node> operator/(const std::shared_ptr<Node> &l, const double &r) {
    return std::make_shared<DivOpNode>(l->value / r, l, std::make_shared<ConstantNode>(r));
}

std::shared_ptr<Node> operator/(const double &l, const std::shared_ptr<Node> &r) {
    return std::make_shared<DivOpNode>(l / r->value, std::make_shared<ConstantNode>(l), r);
}

std::shared_ptr<Node> operator/(const Variable &l, const Variable &r) {
    return l.VarNodePtr / r.VarNodePtr;
}

std::shared_ptr<Node> operator/(const Variable &l, const std::shared_ptr<Node> &r) {
    return l.VarNodePtr / r;
}

std::shared_ptr<Node> operator/(const std::shared_ptr<Node> &l, const Variable &r) {
    return l / r.VarNodePtr;
}

std::shared_ptr<Node> operator/(const double &l, const Variable &r) {
    return l / r.VarNodePtr;
}

std::shared_ptr<Node> operator/(const Variable &l, const double &r) {
    return l.VarNodePtr / r;
}

}  // namespace autodiff
