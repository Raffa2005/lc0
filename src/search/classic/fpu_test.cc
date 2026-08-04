/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "search/classic/fpu.h"

#include <gtest/gtest.h>

namespace lczero {
namespace classic {
namespace {

TEST(FirstPlayUrgencyTest, OrdinaryNodeUsesOpponentOfPreviousMover) {
  EXPECT_FLOAT_EQ(ComputeFirstPlayUrgency(0.0f, false, 0.7f, 0.0f, 0.0f,
                                          0.0f, false),
                  -0.7f);
  EXPECT_FLOAT_EQ(ComputeFirstPlayUrgency(0.0f, false, -0.7f, 0.0f, 0.0f,
                                          0.0f, false),
                  0.7f);
}

TEST(FirstPlayUrgencyTest, MidDoubleNodeKeepsPreviousMoverPerspective) {
  EXPECT_FLOAT_EQ(ComputeFirstPlayUrgency(0.0f, false, 0.7f, 0.0f, 0.0f,
                                          0.0f, true),
                  0.7f);
  EXPECT_FLOAT_EQ(ComputeFirstPlayUrgency(0.0f, false, -0.7f, 0.0f, 0.0f,
                                          0.0f, true),
                  -0.7f);
}

TEST(FirstPlayUrgencyTest, ReductionAppliesAfterPerspectiveConversion) {
  EXPECT_FLOAT_EQ(ComputeFirstPlayUrgency(0.2f, false, 0.4f, 0.0f, 0.0f,
                                          0.25f, false),
                  -0.5f);
  EXPECT_FLOAT_EQ(ComputeFirstPlayUrgency(0.2f, false, 0.4f, 0.0f, 0.0f,
                                          0.25f, true),
                  0.3f);
}

TEST(FirstPlayUrgencyTest, DrawUtilityDoesNotChangeWithMoverPerspective) {
  EXPECT_FLOAT_EQ(ComputeFirstPlayUrgency(0.0f, false, 0.6f, 0.5f, 0.2f,
                                          0.0f, false),
                  -0.5f);
  EXPECT_FLOAT_EQ(ComputeFirstPlayUrgency(0.0f, false, 0.6f, 0.5f, 0.2f,
                                          0.0f, true),
                  0.7f);
}

TEST(FirstPlayUrgencyTest, AbsoluteStrategyIgnoresPerspective) {
  for (const bool same_player : {false, true}) {
    EXPECT_FLOAT_EQ(ComputeFirstPlayUrgency(-0.25f, true, 0.7f, 0.1f, 0.2f,
                                            0.81f, same_player),
                    -0.25f);
  }
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
