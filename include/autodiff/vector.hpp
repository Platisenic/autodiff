#pragma once

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <autodiff/variable.hpp>
#include <autodiff/mathfunctions.hpp>

namespace autodiff {

class Vector {
 public:
    Vector(size_t nsize)
      : m_size(nsize) {
        if (size()) {
            m_buffer = new Variable[size()];
        } else {
            m_buffer = nullptr;
        }
    }
    Vector(std::vector<double> &v)
      : m_size(v.size()) {
          if (size()) {
              m_buffer = new Variable[size()];
              for (size_t i=0; i < v.size(); i++) {
                  m_buffer[i] = v[i];
              }
          } else {
              m_buffer = nullptr;
          }
    }

    size_t size() const { return m_size; }

    std::vector<double> grad() {
        std::vector<double> gradients;
        gradients.reserve(size());
        for (size_t i=0; i < size(); i++) {
            gradients.push_back((*this)(i).VarNodePtr->getGradient());
        }
        return gradients;
    }

    std::vector<double> values() {
        std::vector<double> value;
        value.reserve(size());
        for (size_t i=0; i < size(); i++) {
            value.push_back((*this)(i).values());
        }
        return value;
    }

    void backward() {
        for (size_t i=0; i < size(); i++) {
            (*this)(i).VarNodePtr->prop(1.0);
        }
    }
    double getitem(int index) {
        if (index < 0 || index >= static_cast<int>(size())) throw std::runtime_error("index out of range");
        return m_buffer[index].values();
    }

    void setitem(int index, double value) {
        if (index < 0 || index >= static_cast<int>(size())) throw std::runtime_error("index out of range");
        m_buffer[index] = value;
    }

    std::string info() {
        std::string res;
        res += "[ ";
        for (size_t i=0; i < size(); i++) {
            res += std::to_string((*this)(i).values());
            res += " ";
        }
        res += "]";
        return res;
    }

    Variable   operator() (size_t index) const { return m_buffer[index]; }
    Variable & operator() (size_t index)       { return m_buffer[index]; }

    Variable   operator[] (size_t index) const { return m_buffer[index]; }
    Variable & operator[] (size_t index)       { return m_buffer[index]; }

    Vector operator+(const Vector &r) const {
        if (r.size() != size()) throw std::runtime_error( "size not same" );
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res(i) = (*this)(i) + r(i);
        }
        return res;
    }

    Vector operator+(const double &r) const {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res(i) = (*this)(i) + r;
        }
        return res;
    }

    friend Vector operator+(double l, const Vector &r) {
        Vector res(r.size());
        for (size_t i=0; i < r.size(); i++) {
            res(i) = l + r(i);
        }
        return res;
    }

    Vector operator-(const Vector &r) const {
        if (r.size() != size()) throw std::runtime_error( "size not same" );
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res(i) = (*this)(i) - r(i);
        }
        return res;
    }

    Vector operator-(const double &r) const {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res(i) = (*this)(i) - r;
        }
        return res;
    }

    friend Vector operator-(double l, const Vector &r) {
        Vector res(r.size());
        for (size_t i=0; i < r.size(); i++) {
            res(i) = l - r(i);
        }
        return res;
    }

    Vector operator*(const Vector &r) const {
        if (r.size() != size()) throw std::runtime_error( "size not same" );
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res(i) = (*this)(i) * r(i);
        }
        return res;
    }

    Vector operator*(const double &r) const {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res(i) = (*this)(i) * r;
        }
        return res;
    }

    friend Vector operator*(double l, const Vector &r) {
        Vector res(r.size());
        for (size_t i=0; i < r.size(); i++) {
            res(i) = l * r(i);
        }
        return res;
    }

    Vector operator/(const Vector &r) const {
        if (r.size() != size()) throw std::runtime_error( "size not same" );
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res(i) = (*this)(i) / r(i);
        }
        return res;
    }

    Vector operator/(const double &r) const {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res(i) = (*this)(i) / r;
        }
        return res;
    }

    friend Vector operator/(double l, const Vector &r) {
        Vector res(r.size());
        for (size_t i=0; i < r.size(); i++) {
            res(i) = l / r(i);
        }
        return res;
    }

    Vector sin() {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res[i] = autodiff::sin((*this)(i).VarNodePtr);
        }
        return res;
    }

    Vector cos() {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res[i] = autodiff::cos((*this)(i).VarNodePtr);
        }
        return res;
    }

    Vector tan() {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res[i] = autodiff::tan((*this)(i).VarNodePtr);
        }
        return res;
    }

    Vector exp() {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res[i] = autodiff::exp((*this)(i).VarNodePtr);
        }
        return res;
    }

    Vector log() {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res[i] = autodiff::log((*this)(i).VarNodePtr);
        }
        return res;
    }

    Vector sqrt() {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res[i] = autodiff::sqrt((*this)(i).VarNodePtr);
        }
        return res;
    }

    Vector abs() {
        Vector res(size());
        for (size_t i=0; i < size(); i++) {
            res[i] = autodiff::abs((*this)(i).VarNodePtr);
        }
        return res;
    }

 private:
    size_t m_size = 0;
    Variable * m_buffer = nullptr;
};

}  // namespace autodiff
