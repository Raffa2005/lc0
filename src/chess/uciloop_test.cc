/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
*/

#include "chess/uciloop.h"

#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "chess/gamestate.h"
#include "chess/position.h"

namespace lczero {
namespace {

class CapturingResponder final : public StringUciResponder {
 public:
  void SendRawResponses(const std::vector<std::string>& responses) override {
    responses_ = responses;
  }

  const std::vector<std::string>& responses() const { return responses_; }

 private:
  std::vector<std::string> responses_;
};

class StatefulTestEngine final : public EngineControllerBase {
 public:
  StatefulTestEngine()
      : state_(MakeGameStateFromUci(ChessBoard::kStartposFen, {})) {}

  void EnsureReady() override {}
  void NewGame() override {
    state_ = MakeGameStateFromUci(ChessBoard::kStartposFen, {});
  }
  void SetPosition(const std::string& fen,
                   const std::vector<std::string>& moves) override {
    GameState next = MakeGameStateFromUci(fen, moves);
    state_ = std::move(next);
  }
  GameState GetGameState() const override { return state_; }
  void Go(const GoParams&) override {}
  void PonderHit() override {}
  void Wait() override {}
  void Stop() override {}
  void RegisterUciResponder(UciResponder*) override {}
  void UnregisterUciResponder(UciResponder*) override {}

 private:
  GameState state_;
};

class DmInfoUciTest : public ::testing::Test {
 protected:
  DmInfoUciTest() : loop_(&responder_, &options_, &engine_) {}

  void Process(const std::string& command) {
    ASSERT_TRUE(loop_.ProcessLine(command));
  }

  bool LegalLineContains(std::string_view expected) const {
    if (responder_.responses().size() != 6) return false;
    std::istringstream stream(responder_.responses()[4]);
    std::string token;
    stream >> token >> token;
    while (stream >> token) {
      if (token == expected) return true;
    }
    return false;
  }

  OptionsParser options_;
  CapturingResponder responder_;
  StatefulTestEngine engine_;
  UciLoop loop_;
};

TEST_F(DmInfoUciTest, ReportsStartPositionBeforeAnyPositionCommand) {
  Process("dminfo");

  ASSERT_EQ(responder_.responses().size(), 6);
  EXPECT_EQ(responder_.responses()[0],
            std::string("dminfo fen ") + ChessBoard::kStartposFen);
  EXPECT_EQ(responder_.responses()[1], "dminfo turn w");
  EXPECT_EQ(responder_.responses()[2], "dminfo ability 1 1 0");
  EXPECT_EQ(responder_.responses()[3], "dminfo result ongoing");
  EXPECT_TRUE(LegalLineContains("e2e4"));
  EXPECT_TRUE(LegalLineContains("e2e4+"));
  EXPECT_EQ(responder_.responses()[5], "dminfo ready");
}

TEST_F(DmInfoUciTest, ReportsMidDoubleMoveWithoutAbilityAlternatives) {
  Process("position startpos moves e2e4+");
  Process("dminfo");

  ASSERT_EQ(responder_.responses().size(), 6);
  EXPECT_EQ(responder_.responses()[0],
            "dminfo fen "
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR "
            "w KQkq - 0 1 -Bm");
  EXPECT_EQ(responder_.responses()[1], "dminfo turn w");
  EXPECT_EQ(responder_.responses()[2], "dminfo ability 0 1 1");
  EXPECT_EQ(responder_.responses()[3], "dminfo result ongoing");
  EXPECT_TRUE(LegalLineContains("e4e5"));
  EXPECT_EQ(responder_.responses()[4].find('+'), std::string::npos);
  EXPECT_EQ(responder_.responses()[5], "dminfo ready");
}

TEST_F(DmInfoUciTest, EmitsBlackMovesInGlobalCoordinatesExactlyOnce) {
  Process("position startpos moves e2e4+ e4e5");
  Process("dminfo");

  EXPECT_EQ(responder_.responses()[1], "dminfo turn b");
  EXPECT_EQ(responder_.responses()[2], "dminfo ability 0 1 0");
  EXPECT_TRUE(LegalLineContains("a7a6"));
  EXPECT_TRUE(LegalLineContains("a7a6+"));
  EXPECT_FALSE(LegalLineContains("a2a3"));
}

TEST_F(DmInfoUciTest, EmitsBlackMidMoveAndPromotionInGlobalCoordinates) {
  Process("position startpos moves e2e4 a7a6+");
  Process("dminfo");
  EXPECT_EQ(responder_.responses()[1], "dminfo turn b");
  EXPECT_EQ(responder_.responses()[2], "dminfo ability 1 0 1");
  EXPECT_TRUE(LegalLineContains("a6a5"));
  EXPECT_EQ(responder_.responses()[4].find('+'), std::string::npos);

  Process("position fen 4k3/8/8/8/8/8/p7/4K3 b - - 0 1");
  Process("dminfo");
  for (const std::string_view suffix : {"n", "b", "r", "q"}) {
    EXPECT_TRUE(
        LegalLineContains(std::string("a2a1") + std::string(suffix)));
    EXPECT_TRUE(LegalLineContains(std::string("a2a1") +
                                  std::string(suffix) + "+"));
  }
}

TEST_F(DmInfoUciTest, EmitsStandardCastlingForBothColors) {
  constexpr char kCastleFen[] =
      "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
  Process(std::string("position fen ") + kCastleFen);
  Process("dminfo");
  EXPECT_TRUE(LegalLineContains("e1g1"));
  EXPECT_TRUE(LegalLineContains("e1c1"));
  EXPECT_TRUE(LegalLineContains("e1g1+"));
  EXPECT_TRUE(LegalLineContains("e1c1+"));

  Process("position fen r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
  Process("dminfo");
  EXPECT_TRUE(LegalLineContains("e8g8"));
  EXPECT_TRUE(LegalLineContains("e8c8"));
  EXPECT_TRUE(LegalLineContains("e8g8+"));
  EXPECT_TRUE(LegalLineContains("e8c8+"));
}

TEST_F(DmInfoUciTest, TerminalPositionHasEmptyLegalFrameAndReadySentinel) {
  Process("position fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1 --");
  Process("dminfo");

  ASSERT_EQ(responder_.responses().size(), 6);
  EXPECT_EQ(responder_.responses()[3], "dminfo result white");
  EXPECT_EQ(responder_.responses()[4], "dminfo legal");
  EXPECT_EQ(responder_.responses()[5], "dminfo ready");
}

TEST_F(DmInfoUciTest, InvalidPositionDoesNotReplaceLastValidState) {
  Process("position startpos moves e2e4");
  Process("dminfo");
  const auto valid_responses = responder_.responses();

  EXPECT_THROW(loop_.ProcessLine("position startpos moves e2e4 e7e4"),
               Exception);
  Process("dminfo");
  EXPECT_EQ(responder_.responses(), valid_responses);
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
