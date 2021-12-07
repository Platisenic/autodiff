#pragma once

#include <cmath>
#include <memory>
#include <autodiff/node.hpp>

namespace autodiff {

std::shared_ptr<Node> sin(const std::shared_ptr<Node> &l) {
    return std::make_shared<SinOpNode>(std::sin(l->value), l);
}

std::shared_ptr<Node> cos(const std::shared_ptr<Node> &l) {
    return std::make_shared<CosOpNode>(std::cos(l->value), l);
}

std::shared_ptr<Node> tan(const std::shared_ptr<Node> &l) {
    return std::make_shared<TanOpNode>(std::tan(l->value), l);
}

std::shared_ptr<Node> exp(const std::shared_ptr<Node> &l) {
    return std::make_shared<ExpOpNode>(std::exp(l->value), l);
}

std::shared_ptr<Node> log(const std::shared_ptr<Node> &l) {
    return std::make_shared<LogOpNode>(std::log(l->value), l);
}

std::shared_ptr<Node> sqrt(const std::shared_ptr<Node> &l) {
    return std::make_shared<SqrtOpNode>(std::sqrt(l->value), l);
}

std::shared_ptr<Node> abs(const std::shared_ptr<Node> &l) {
    return std::make_shared<AbsOpNode>(std::abs(l->value), l);
}

}  // namespace autodiff
