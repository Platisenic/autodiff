#pragma once

#include <memory>
#include <cmath>

namespace autodiff{

    struct Node{
        double value;

        Node(const double &v): value(v) {}
        ~Node() {}
        virtual void prop(const double &output) = 0;
    };

    struct ConstantNode: Node{
        ConstantNode(const double &v) : Node(v) {}
        void prop(const double & /*output*/) override { /* do nothing */ }
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
            const std::shared_ptr<Node> &l,
            const std::shared_ptr<Node> &r) :
            Node(v),
            left(l),
            right(r)
            {}
    };

    struct AddOpNode: BinaryOpNode{
        AddOpNode(const double &v,
            const std::shared_ptr<Node> &l,
            const std::shared_ptr<Node> &r) :
            BinaryOpNode(v, l, r)
            {}

        void prop(const double &output) override{
            left->prop(output);
            right->prop(output);
        }
    };

    struct SubOpNode: BinaryOpNode{
        SubOpNode(const double &v,
            const std::shared_ptr<Node> &l,
            const std::shared_ptr<Node> &r) :
            BinaryOpNode(v, l, r)
            {}

        void prop(const double &output) override{
            left->prop(output);
            right->prop(-output);
        }
    };

    struct MulOpNode: BinaryOpNode{
        MulOpNode(const double &v,
            const std::shared_ptr<Node> &l,
            const std::shared_ptr<Node> &r) :
            BinaryOpNode(v, l, r)
            {}

        void prop(const double &output) override{
            left->prop(right->value * output);
            right->prop(left->value * output);
        }
    };

    struct DivOpNode: BinaryOpNode{
        DivOpNode(const double &v,
            const std::shared_ptr<Node> &l,
            const std::shared_ptr<Node> &r) :
            BinaryOpNode(v, l, r)
            {}

        void prop(const double &output) override{
            double recRight = 1.0 / right->value;
            left->prop(recRight * output);
            right->prop(recRight * recRight * (-left->value) * output);
        }
    };

    struct UnaryOpNode: Node{
        std::shared_ptr<Node> m;

        UnaryOpNode(const double &v,
            const std::shared_ptr<Node> m) :
            Node(v),
            m(m)
            {}
    };

    struct NegOpNode: UnaryOpNode{
        NegOpNode(const double &v,
            const std::shared_ptr<Node> m) :
            UnaryOpNode(v, m)
            {}
        void prop(const double &output) override{
            m->prop(-output);
        }
    };

    struct SinOpNode: UnaryOpNode{
        SinOpNode(const double &v,
            const std::shared_ptr<Node> m) :
            UnaryOpNode(v, m)
            {}
        void prop(const double &output) override{
            m->prop(std::cos(m->value) * output);
        }
    };

    struct CosOpNode: UnaryOpNode{
        CosOpNode(const double &v,
            const std::shared_ptr<Node> m) :
            UnaryOpNode(v, m)
            {}
        void prop(const double &output) override{
            m->prop(std::sin(m->value) * (-output));
        }
    };

    struct TanOpNode: UnaryOpNode{
        TanOpNode(const double &v,
            const std::shared_ptr<Node> m) :
            UnaryOpNode(v, m)
            {}
        void prop(const double &output) override{
            double secx = 1.0 / std::cos(m->value);
            m->prop(secx * secx * output);
        }
    };

    struct ExpOpNode: UnaryOpNode{
        ExpOpNode(const double &v,
            const std::shared_ptr<Node> m) :
            UnaryOpNode(v, m)
            {}
        void prop(const double &output) override{
            m->prop(std::exp(m->value) * output);
        }
    };

    struct LogOpNode: UnaryOpNode{
        LogOpNode(const double &v,
            const std::shared_ptr<Node> m) :
            UnaryOpNode(v, m)
            {}
        void prop(const double &output) override{
            m->prop(output / m->value);
        }
    };

    struct SqrtOpNode: UnaryOpNode{
        SqrtOpNode(const double &v,
            const std::shared_ptr<Node> m) :
            UnaryOpNode(v, m)
            {}
        void prop(const double &output) override{
            m->prop(output / (2.0 * std::sqrt(m->value)));
        }
    };

    struct AbsOpNode: UnaryOpNode{
        AbsOpNode(const double &v,
            const std::shared_ptr<Node> m) :
            UnaryOpNode(v, m)
            {}
        void prop(const double &output) override{
            if(m->value > 0.0){
                m->prop(output);
            }else if (m->value < 0.0){
                m->prop(-output);
            }else{
                m->prop(0.0);
            }
        }
    };
}
