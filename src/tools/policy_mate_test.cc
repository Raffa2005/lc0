// SPDX-License-Identifier: GPL-3.0-or-later
#include "tools/policy_mate.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <sstream>

#include "search/classic/node.h"

namespace lczero {
namespace {
class PolicyMateTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { InitializeMagicBitboards(); }
};

std::set<std::string> MateLines(const PolicyMateExpansion& expansion) {
  std::set<std::string> lines;
  for (const auto& state : expansion.mates) {
    const auto uci = PolicyMateUci(state);
    lines.insert(uci);
    std::istringstream stream(uci);
    std::vector<std::string> moves;
    std::string token;
    while (stream >> token) moves.push_back(token);
    EXPECT_EQ(MakeGameStateFromUci(PositionToFen(state.startpos), moves)
                  .GetPositionHistory().ComputeGameResult(), GameResult::WHITE_WON);
  }
  return lines;
}

TEST_F(PolicyMateTest, KnownV42WitnessAndNoMidTurnTermination) {
  const auto state = MakeGameStateFromUci(ChessBoard::kStartposFen,
      {"e2e4", "d7d5", "e4e5", "c7c5", "d1h5", "b8c6"});
  const auto lines = MateLines(ExpandPolicyMateWhite(state, false));
  EXPECT_TRUE(lines.contains("e2e4 d7d5 e4e5 c7c5 d1h5 b8c6 e5e6+ h5f7"));
  auto history = state.GetPositionHistory();
  history.Append(history.Last().GetBoard().ParseMove("e5e6"), true);
  EXPECT_EQ(history.ComputeGameResult(), GameResult::UNDECIDED);
}

TEST_F(PolicyMateTest, EnumerationMatchesPublicLegalMoveOracle) {
  for (const std::string fen : {
      "7k/8/5KQ1/8/8/8/8/8 w - - 0 1 WB",
      "k7/8/8/8/8/8/8/1R5K w - - 0 1 WB",
      "4k3/P7/8/8/8/8/8/4K3 w - - 0 1 WB",
      "7r/8/7P/8/1n6/8/2k5/K7 w - - 0 1 WB"}) {
    const auto state = MakeGameStateFromUci(fen, {});
    std::set<std::string> expected;
    uint64_t pairs = 0;
    for (auto first : GenerateDoubleMoveLegalMoves(state)) {
      auto after_first = state;
      after_first.moves.push_back(first);
      if (!first.is_ability_activation()) {
        if (after_first.GetPositionHistory().ComputeGameResult() == GameResult::WHITE_WON) {
          expected.insert(PolicyMateUci(after_first));
        }
        continue;
      }
      for (auto second : GenerateDoubleMoveLegalMoves(after_first)) {
        ++pairs;
        auto after_second = after_first;
        after_second.moves.push_back(second);
        if (after_second.GetPositionHistory().ComputeGameResult() == GameResult::WHITE_WON) {
          expected.insert(PolicyMateUci(after_second));
        }
      }
    }
    const auto actual = ExpandPolicyMateWhite(state, true);
    EXPECT_EQ(MateLines(actual), expected) << fen;
    EXPECT_EQ(actual.double_pairs, pairs) << fen;
  }
}

TEST_F(PolicyMateTest, ArgmaxUsesCompressedPriorsAndClassicTieOrder) {
  auto legal = GenerateDoubleMoveLegalMoves(MakeGameStateFromUci(ChessBoard::kStartposFen, {}));
  legal.resize(3);
  EXPECT_EQ(PolicyMateBest(std::vector<float>{0.1f, 0.8f, 0.1f}, legal), 1);
  legal.resize(2);
  EXPECT_EQ(PolicyMateBest(std::vector<float>{0.5f, 0.5f}, legal), 0);
  // These differ before compression but occupy the same search edge bin.
  classic::Edge a, b;
  a.SetP(0.5f); b.SetP(0.500001f);
  ASSERT_EQ(a.GetP(), b.GetP());
  EXPECT_EQ(PolicyMateBest(std::vector<float>{0.5f, 0.500001f}, legal), 0);
  EXPECT_THROW(PolicyMateBest(std::vector<float>{}, legal), Exception);
  EXPECT_THROW(PolicyMateBest(std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 1}, legal), Exception);

  legal = GenerateDoubleMoveLegalMoves(MakeGameStateFromUci(ChessBoard::kStartposFen, {}));
  classic::Node node(nullptr, 0);
  node.CreateEdges(legal);
  for (auto& edge : node.Edges()) edge.edge()->SetP(0.025f);
  node.SortEdges();
  for (auto& edge : node.Edges()) {
    EXPECT_EQ(legal[PolicyMateBest(std::vector<float>(legal.size(), 0.025f), legal)], edge.GetMove());
    break;
  }
}

TEST_F(PolicyMateTest, BlackAbilityIsPrunedNotMaskedAndInferenceIsBatched) {
  const auto start = MakeGameStateFromUci(ChessBoard::kStartposFen, {});
  std::vector<size_t> sizes;
  auto evaluate = [&](std::span<const GameState> states, const std::vector<MoveList>& legal) {
    sizes.push_back(states.size());
    std::vector<PolicyMateDecision> result;
    for (const auto& moves : legal) {
      std::vector<float> p(moves.size());
      const auto chosen = std::find_if(moves.begin(), moves.end(),
                                      [](auto m) { return m.is_ability_activation(); });
      EXPECT_NE(chosen, moves.end());
      p[chosen - moves.begin()] = 1;
      result.push_back({p, PolicyMateBest(p, moves)});
    }
    return result;
  };
  std::ostringstream output, trace;
  SearchPolicyMates(start, {.white_turns = 3, .threads = 4, .batch_size = 7},
                    evaluate, output, trace);
  EXPECT_EQ(sizes, (std::vector<size_t>{7, 7, 6}));
  EXPECT_NE(output.str().find("\"pruned_black_activation\":20"), std::string::npos);
  EXPECT_NE(output.str().find("restricted_tree_exhausted"), std::string::npos);
  EXPECT_EQ(output.str().find("\"type\":\"mate\""), std::string::npos);
}

TEST_F(PolicyMateTest, ThreadsPreserveEnumerationAndBlackTrace) {
  const auto start = MakeGameStateFromUci(ChessBoard::kStartposFen, {});
  auto evaluate = [](std::span<const GameState>, const std::vector<MoveList>& legal) {
    std::vector<PolicyMateDecision> result;
    for (const auto& moves : legal) {
      std::vector<float> p(moves.size(), 0);
      p[0] = 1;
      result.push_back({p, PolicyMateBest(p, moves)});
    }
    return result;
  };
  std::ostringstream one, four, trace_one, trace_four;
  SearchPolicyMates(start, {.white_turns = 2, .threads = 1}, evaluate, one, trace_one);
  SearchPolicyMates(start, {.white_turns = 2, .threads = 4}, evaluate, four, trace_four);
  EXPECT_EQ(trace_one.str(), trace_four.str());
  for (auto* stream : {&one, &four}) {
    EXPECT_NE(stream->str().find("\"complete_white_turns\":2"), std::string::npos);
    EXPECT_EQ(stream->str().find("\"type\":\"mate\""), std::string::npos);
  }
}

TEST_F(PolicyMateTest, FrontierLimitDoesNotClaimNextDepthComplete) {
  auto evaluate = [](std::span<const GameState>, const std::vector<MoveList>&) {
    ADD_FAILURE() << "Should stop before inference";
    return std::vector<PolicyMateDecision>{};
  };
  std::ostringstream output, trace;
  SearchPolicyMates(MakeGameStateFromUci(ChessBoard::kStartposFen, {}),
      {.white_turns = 3, .max_frontier = 1}, evaluate, output, trace);
  EXPECT_NE(output.str().find("\"status\":\"frontier_limit\""), std::string::npos);
  EXPECT_NE(output.str().find("\"complete_white_turns\":1"), std::string::npos);
}

TEST_F(PolicyMateTest, PruningUsesSearchedMoveInsteadOfTopPrior) {
  auto evaluate = [](std::span<const GameState>, const std::vector<MoveList>& legal) {
    std::vector<PolicyMateDecision> result;
    for (const auto& moves : legal) {
      std::vector<float> p(moves.size(), 0);
      const auto ordinary = std::find_if(moves.begin(), moves.end(),
          [](auto m) { return !m.is_ability_activation(); });
      const auto ability = std::find_if(moves.begin(), moves.end(),
          [](auto m) { return m.is_ability_activation(); });
      p[ordinary - moves.begin()] = 1;
      result.push_back({p, static_cast<size_t>(ability - moves.begin()), 25});
    }
    return result;
  };
  std::ostringstream output, trace;
  SearchPolicyMates(MakeGameStateFromUci(ChessBoard::kStartposFen, {}),
      {.white_turns = 2, .black_visits = 25}, evaluate, output, trace);
  EXPECT_NE(output.str().find("\"pruned_black_activation\":20"), std::string::npos);
  EXPECT_NE(output.str().find("\"black_policy\":\"classic_mcts\""), std::string::npos);
  EXPECT_NE(trace.str().find("\t1\t25\n"), std::string::npos);
}
}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
