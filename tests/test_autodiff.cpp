#include <gtest/gtest.h>
#include <autodiff/autodiff.hpp>
#include <vector>
#include <cmath>

using namespace autodiff;

TEST(AutoDiffTest, AddOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  auto b = std::make_shared<IndVarNode>(2.0);
  auto a_b_add = a + b;
  a_b_add->prop(1.0);
  EXPECT_NEAR(a_b_add->value, 3.0, 1e-10);
  EXPECT_NEAR(a->grad, 1.0, 1e-10);
  EXPECT_NEAR(b->grad, 1.0, 1e-10);
}

TEST(AutoDiffTest, SubOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  auto b = std::make_shared<IndVarNode>(2.0);
  auto a_b_sub = a - b;
  a_b_sub->prop(1.0);
  EXPECT_NEAR(a_b_sub->value, -1.0, 1e-10);
  EXPECT_NEAR(a->grad, 1.0, 1e-10);
  EXPECT_NEAR(b->grad, -1.0, 1e-10);
}

TEST(AutoDiffTest, MulOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  auto b = std::make_shared<IndVarNode>(2.0);
  auto a_b_mul = a * b;
  a_b_mul->prop(1.0);
  EXPECT_NEAR(a_b_mul->value, 2.0, 1e-10);
  EXPECT_NEAR(a->grad, 2.0, 1e-10);
  EXPECT_NEAR(b->grad, 1.0, 1e-10);
}

TEST(AutoDiffTest, DivOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  auto b = std::make_shared<IndVarNode>(2.0);
  auto a_b_div = a / b;
  a_b_div->prop(1.0);
  EXPECT_NEAR(a_b_div->value, 0.5, 1e-10);
  EXPECT_NEAR(a->grad, 0.5, 1e-10);
  EXPECT_NEAR(b->grad, -0.25, 1e-10);
}


TEST(AutoDiffTest, ConstantNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  double b = 2.0;
  auto a_b_add = a + b;
  a_b_add->prop(1.0);
  EXPECT_NEAR(a_b_add->value, 3.0, 1e-10);
  EXPECT_NEAR(a->grad, 1.0, 1e-10);
  a->grad = 0.0;

  auto a_b_sub = a - b;
  a_b_sub->prop(1.0);
  EXPECT_NEAR(a_b_sub->value, -1.0, 1e-10);
  EXPECT_NEAR(a->grad, 1.0, 1e-10);
  a->grad = 0.0;

  auto a_b_mul = a * b;
  a_b_mul->prop(1.0);
  EXPECT_NEAR(a_b_mul->value, 2.0, 1e-10);
  EXPECT_NEAR(a->grad, 2.0, 1e-10);
  a->grad = 0.0;

  auto a_b_div = a / b;
  a_b_div->prop(1.0);
  EXPECT_NEAR(a_b_div->value, 0.5, 1e-10);
  EXPECT_NEAR(a->grad, 0.5, 1e-10);
  a->grad = 0.0;
}


TEST(AutoDiffTest, NegOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(3.0);
  auto a_b_neg = -a;
  a_b_neg->prop(1.0);
  EXPECT_NEAR(a_b_neg->value, -3.0, 1e-10);
  EXPECT_NEAR(a->grad, -1.0, 1e-10);
}

TEST(AutoDiffTest, SinOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(M_PI);
  auto sin_a = sin(a);
  sin_a->prop(1.0);
  EXPECT_NEAR(sin_a->value, 0, 1e-10);
  EXPECT_NEAR(a->grad, -1.0, 1e-10);
}

TEST(AutoDiffTest, CosOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(M_PI);
  auto cos_a = cos(a);
  cos_a->prop(1.0);
  EXPECT_NEAR(cos_a->value, -1.0, 1e-10);
  EXPECT_NEAR(a->grad, 0, 1e-10);
}

TEST(AutoDiffTest, TanOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(M_PI);
  auto tan_a = tan(a);
  tan_a->prop(1.0);
  EXPECT_NEAR(tan_a->value, 0, 1e-10);
  EXPECT_NEAR(a->grad, 1.0, 1e-10);
}

TEST(AutoDiffTest, ExpOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(std::log(5.0));
  auto exp_a = exp(a);
  exp_a->prop(1.0);
  EXPECT_NEAR(exp_a->value, 5.0, 1e-10);
  EXPECT_NEAR(a->grad, 5.0, 1e-10);
}

TEST(AutoDiffTest, LogOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(std::exp(5.0));
  auto log_a = log(a);
  log_a->prop(1.0);
  EXPECT_NEAR(log_a->value, 5.0, 1e-10);
  EXPECT_NEAR(a->grad, 1 / std::exp(5.0), 1e-10);
}

TEST(AutoDiffTest, SqrtOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(100.0);
  auto sqrt_a = sqrt(a);
  sqrt_a->prop(1.0);
  EXPECT_NEAR(sqrt_a->value, 10.0, 1e-10);
  EXPECT_NEAR(a->grad, 0.05, 1e-10);
}

TEST(AutoDiffTest, AbsOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(-5.0);
  auto abs_a = abs(a);
  abs_a->prop(1.0);
  EXPECT_NEAR(abs_a->value, 5.0, 1e-10);
  EXPECT_NEAR(a->grad, -1.0, 1e-10);
}

TEST(AutoDiffTest, VariableTest) {
  std::vector<Variable> inputs { Variable(1.0), Variable(2.0), Variable(M_PI) };
  Variable output;
  output = inputs[0] + inputs[1];
  EXPECT_NEAR(output.values(), 3.0, 1e-10);
  output = inputs[0] - inputs[1];
  EXPECT_NEAR(output.values(), -1.0, 1e-10);
  output = inputs[0] * inputs[1];
  EXPECT_NEAR(output.values(), 2.0, 1e-10);
  output = inputs[0] / inputs[1];
  EXPECT_NEAR(output.values(), 0.5, 1e-10);
  output = inputs[0] * inputs[1] + inputs[0];
  EXPECT_NEAR(output.values(), 3.0, 1e-10);
  output = sin(inputs[2]);
  EXPECT_NEAR(output.values(), 0, 1e-10);
}

TEST(AutoDiffTest, VectorTest) {
  Vector m(3); m[0] = 1.0; m[1] = 2.0; m[2] = M_PI;
  Variable output;
  output = m[0] + m[1];
  EXPECT_NEAR(output.values(), 3.0, 1e-10);
  output = m[0] - m[1];
  EXPECT_NEAR(output.values(), -1.0, 1e-10);
  output = m[0] * m[1];
  EXPECT_NEAR(output.values(), 2.0, 1e-10);
  output = m[0] / m[1];
  EXPECT_NEAR(output.values(), 0.5, 1e-10);
  output = m[0] * m[1] + m[0];
  EXPECT_NEAR(output.values(), 3.0, 1e-10);
  output = sin(m[2]);
  EXPECT_NEAR(output.values(), 0, 1e-10);
}

TEST(AutoDiffTest, GradentValueTest1) {
  Vector a(2); a[0] = 1.0; a[1] = 1.0;
  Vector b(2); b[0] = 2.0; b[1] = 2.0;
  Vector c(2); c[0] = M_PI; c[1] = M_PI;

  Vector o = (a * b + c.sin()).exp().log();
  std::vector<double> goldensA { 2.0, 2.0 };
  std::vector<double> goldensB { 1.0, 1.0 };
  std::vector<double> goldensC { -1.0, -1.0 };

  o.backward();

  std::vector<double> testA = a.grad();
  for (size_t i=0; i < testA.size(); i++) {
    EXPECT_NEAR(goldensA[i], testA[i], 1e-10);
  }

  std::vector<double> testB = b.grad();
  for (size_t i=0; i < testA.size(); i++) {
    EXPECT_NEAR(goldensB[i], testB[i], 1e-10);
  }

  std::vector<double> testC = c.grad();
  for (size_t i=0; i < testC.size(); i++) {
    EXPECT_NEAR(goldensC[i], testC[i], 1e-10);
  }

}

TEST(AutoDiffTest, GradentValueTest2) {
  Vector a(2); a[0] = 1.0; a[1] = 1.0;
  Vector b(2); b[0] = 2.0; b[1] = 2.0;
  Vector a_b_mul = a * b;

  std::vector<double> goldensA { 2.0, 2.0 };
  std::vector<double> goldensB { 1.0, 1.0 };

  a_b_mul.backward();

  std::vector<double> testA = a.grad();
  for (size_t i=0; i < testA.size(); i++) {
    EXPECT_NEAR(goldensA[i], testA[i], 1e-10);
  }
  std::vector<double> testB = b.grad();
  for (size_t i=0; i < testB.size(); i++) {
    EXPECT_NEAR(goldensB[i], testB[i], 1e-10);
  }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
