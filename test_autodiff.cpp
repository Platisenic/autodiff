#include <gtest/gtest.h>
#include "autodiff.hpp"

TEST(AutoDiffTest, AddOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  auto b = std::make_shared<IndVarNode>(2.0);
  auto a_b_add = a + b;
  a_b_add->prop(1.0);
  EXPECT_EQ(a_b_add->value, 3.0);
  EXPECT_EQ(a->grad, 1.0);
  EXPECT_EQ(b->grad, 1.0);
}

TEST(AutoDiffTest, SubOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  auto b = std::make_shared<IndVarNode>(2.0);
  auto a_b_sub = a - b;
  a_b_sub->prop(1.0);
  EXPECT_EQ(a_b_sub->value, -1.0);
  EXPECT_EQ(a->grad, 1.0);
  EXPECT_EQ(b->grad, -1.0);
}

TEST(AutoDiffTest, MulOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  auto b = std::make_shared<IndVarNode>(2.0);
  auto a_b_mul = a * b;
  a_b_mul->prop(1.0);
  EXPECT_EQ(a_b_mul->value, 2.0);
  EXPECT_EQ(a->grad, 2.0);
  EXPECT_EQ(b->grad, 1.0);
}

TEST(AutoDiffTest, DivOpNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  auto b = std::make_shared<IndVarNode>(2.0);
  auto a_b_div = a / b;
  a_b_div->prop(1.0);
  EXPECT_EQ(a_b_div->value, 0.5);
  EXPECT_EQ(a->grad, 0.5);
  EXPECT_EQ(b->grad, -0.25);
}


TEST(AutoDiffTest, ConstantNodeTest) {
  auto a = std::make_shared<IndVarNode>(1.0);
  double b = 2.0;
  auto a_b_add = a + b;
  a_b_add->prop(1.0);
  EXPECT_EQ(a_b_add->value, 3.0);
  EXPECT_EQ(a->grad, 1.0);
  a->grad = 0.0;

  auto a_b_sub = a - b;
  a_b_sub->prop(1.0);
  EXPECT_EQ(a_b_sub->value, -1.0);
  EXPECT_EQ(a->grad, 1.0);
  a->grad = 0.0;

  auto a_b_mul = a * b;
  a_b_mul->prop(1.0);
  EXPECT_EQ(a_b_mul->value, 2.0);
  EXPECT_EQ(a->grad, 2.0);
  a->grad = 0.0;

  auto a_b_div = a / b;
  a_b_div->prop(1.0);
  EXPECT_EQ(a_b_div->value, 0.5);
  EXPECT_EQ(a->grad, 0.5);
  a->grad = 0.0;

}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
