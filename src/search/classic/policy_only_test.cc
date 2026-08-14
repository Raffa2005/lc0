/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "search/classic/policy_only.h"

#include <gtest/gtest.h>

namespace lczero {
namespace classic {
namespace {

TEST(PolicyOnlySafePriorRatioTest, KeepsOriginalBandThroughPointFive) {
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(-1.0f), 0.5f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(0.0f), 0.5f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(0.5f), 0.5f);
}

TEST(PolicyOnlySafePriorRatioTest, WidensLinearlyTowardFivePercent) {
  EXPECT_NEAR(PolicyOnlySafePriorRatio(0.75f), 0.275f, 1e-6f);
  EXPECT_NEAR(PolicyOnlySafePriorRatio(0.9f), 0.14f, 1e-6f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(1.0f), 0.05f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(2.0f), 0.05f);
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
