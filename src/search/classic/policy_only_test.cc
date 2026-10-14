/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "search/classic/policy_only.h"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace lczero {
namespace classic {
namespace {

TEST(PolicyOnlySafePriorRatioTest, NoPriorDropAtLosingEndpoint) {
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(-2.0f), 1.0f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(-1.0f), 1.0f);
}

TEST(PolicyOnlySafePriorRatioTest, LosingValuesAllowLinearlyLessPriorDrop) {
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(-0.75f), 0.875f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(-0.5f), 0.75f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(-0.4f), 0.7f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(-0.25f), 0.625f);
}

TEST(PolicyOnlySafePriorRatioTest, PositiveValuesRemainLinearWithoutFlatBand) {
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(0.0f), 0.5f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(0.25f), 0.375f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(0.5f), 0.25f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(0.75f), 0.125f);
}

TEST(PolicyOnlySafePriorRatioTest, FivePercentFloorStartsAtPointNine) {
  EXPECT_NEAR(PolicyOnlySafePriorRatio(0.9f), 0.05f, 1e-6f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(0.95f), 0.05f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(1.0f), 0.05f);
  EXPECT_FLOAT_EQ(PolicyOnlySafePriorRatio(2.0f), 0.05f);
}

TEST(PolicyOnlySafePriorRatioTest, ContinuousAtPiecewiseBoundaries) {
  for (const float q : {-1.0f, 0.0f, 0.5f, 0.9f, 1.0f}) {
    const float below = std::nextafter(q, -std::numeric_limits<float>::infinity());
    const float above = std::nextafter(q, std::numeric_limits<float>::infinity());
    EXPECT_NEAR(PolicyOnlySafePriorRatio(below), PolicyOnlySafePriorRatio(q),
                1e-6f);
    EXPECT_NEAR(PolicyOnlySafePriorRatio(above), PolicyOnlySafePriorRatio(q),
                1e-6f);
  }
}

TEST(PolicyOnlySafePriorRatioTest, ConstantSlopeAcrossZeroAndOldFlatBand) {
  for (int i = -100; i < 89; ++i) {
    const float q = i / 100.0f;
    const float current = PolicyOnlySafePriorRatio(q);
    const float next = PolicyOnlySafePriorRatio(q + 0.01f);
    EXPECT_LT(next, current);
    EXPECT_NEAR(current - next, 0.005f, 1e-6f);
  }
}

TEST(PolicyOnlySafePriorRatioTest, WinningValueAllowsAnOtherwiseRejectedPrior) {
  constexpr float top_prior = 0.6f;
  constexpr float alternative_prior = 0.18f;
  EXPECT_LT(alternative_prior, top_prior * PolicyOnlySafePriorRatio(0.0f));
  EXPECT_GE(alternative_prior, top_prior * PolicyOnlySafePriorRatio(0.5f));
}

TEST(PolicyOnlySafePriorRatioTest, RatioIsBoundedAndMonotonic) {
  float previous = PolicyOnlySafePriorRatio(-2.0f);
  for (int i = -2000; i <= 2000; ++i) {
    const float current = PolicyOnlySafePriorRatio(i / 1000.0f);
    EXPECT_GE(current, 0.05f);
    EXPECT_LE(current, 1.0f);
    EXPECT_LE(current, previous);
    previous = current;
  }
}

TEST(PolicyOnlySafePriorRatioTest, Rd1MustNotReplaceKh6AtObservedLosingValue) {
  // Rounded priors from the full-history BT3 one-visit reproduction. This
  // tests the generic cutoff; neither move nor position enters production code.
  constexpr float kh6_prior = 0.5308f;
  constexpr float rd1_prior = 0.3590f;
  EXPECT_GE(rd1_prior, kh6_prior * PolicyOnlySafePriorRatio(0.0f));
  EXPECT_LT(rd1_prior, kh6_prior * PolicyOnlySafePriorRatio(-0.3991f));
  EXPECT_GE(kh6_prior, kh6_prior * PolicyOnlySafePriorRatio(-1.0f));
  EXPECT_LT(std::nextafter(kh6_prior, 0.0f),
            kh6_prior * PolicyOnlySafePriorRatio(-1.0f));
}

}  // namespace
}  // namespace classic
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
