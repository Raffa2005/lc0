/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "chess/position.h"

#include <algorithm>
#include <gtest/gtest.h>

#include <iostream>
#include <string_view>
#include <utility>

#include "chess/gamestate.h"
#include "chess/pgn.h"
#include "utils/string.h"

namespace lczero {
namespace {

bool ContainsMove(const MoveList& moves, std::string_view move_string) {
  return std::any_of(moves.begin(), moves.end(), [&](const Move& move) {
    return move.ToString(/*is_chess960=*/false) == move_string;
  });
}

}  // namespace

TEST(GameState, ParsesDoubleMoveUciSequence) {
  const auto state = MakeGameStateFromUci(
      ChessBoard::kStartposFen, {"e2e4+", "g1f3", "e7e5"});

  ASSERT_EQ(state.moves.size(), 3);
  EXPECT_TRUE(state.moves[0].is_ability_activation());
  EXPECT_FALSE(state.moves[1].is_ability_activation());
  EXPECT_FALSE(state.moves[2].is_ability_activation());
  EXPECT_FALSE(state.CurrentPosition().IsBlackToMove());
  EXPECT_FALSE(state.CurrentPosition().IsMidDoublemove());
  EXPECT_FALSE(state.CurrentPosition().OurDoublemoveAvailable());
  EXPECT_TRUE(state.CurrentPosition().TheirDoublemoveAvailable());
}

TEST(GameState, PreservesMidDoubleMoveAtSequenceEnd) {
  const auto state =
      MakeGameStateFromUci(ChessBoard::kStartposFen, {"e2e4+"});

  ASSERT_EQ(state.moves.size(), 1);
  EXPECT_FALSE(state.CurrentPosition().IsBlackToMove());
  EXPECT_TRUE(state.CurrentPosition().IsMidDoublemove());
}

TEST(GameState, AllowsPawnToMoveOnBothHalves) {
  const auto state =
      MakeGameStateFromUci(ChessBoard::kStartposFen, {"e2e4+", "e4e5"});

  EXPECT_FALSE(state.CurrentPosition().IsMidDoublemove());
  EXPECT_TRUE(PositionToFen(state.CurrentPosition()).starts_with(
      "rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR b "));
}

TEST(GameState, GeneratesOrdinaryAndAbilityMovesOnlyWhenAvailable) {
  const auto start = MakeGameStateFromUci(ChessBoard::kStartposFen, {});
  const MoveList start_moves = GenerateDoubleMoveLegalMoves(start);
  EXPECT_EQ(start_moves.size(), 40);
  EXPECT_TRUE(ContainsMove(start_moves, "e2e4"));
  EXPECT_TRUE(ContainsMove(start_moves, "e2e4+"));

  const auto mid =
      MakeGameStateFromUci(ChessBoard::kStartposFen, {"e2e4+"});
  const MoveList mid_moves = GenerateDoubleMoveLegalMoves(mid);
  EXPECT_TRUE(ContainsMove(mid_moves, "e4e5"));
  EXPECT_FALSE(std::any_of(mid_moves.begin(), mid_moves.end(),
                           [](const Move& move) {
                             return move.is_ability_activation();
                           }));

  const auto spent = MakeGameStateFromUci(
      ChessBoard::kStartposFen, {"e2e4+", "e4e5", "a7a6"});
  const MoveList spent_moves = GenerateDoubleMoveLegalMoves(spent);
  EXPECT_TRUE(ContainsMove(spent_moves, "g1f3"));
  EXPECT_FALSE(ContainsMove(spent_moves, "g1f3+"));
}

TEST(GameState, RejectsIllegalAndInvalidAbilityMoves) {
  EXPECT_THROW(
      MakeGameStateFromUci(ChessBoard::kStartposFen, {"e2e5"}), Exception);
  EXPECT_THROW(
      MakeGameStateFromUci(ChessBoard::kStartposFen, {"e2e4q"}), Exception);
  EXPECT_THROW(
      MakeGameStateFromUci(ChessBoard::kStartposFen, {"e2e4++"}), Exception);
  EXPECT_THROW(MakeGameStateFromUci(ChessBoard::kStartposFen,
                                    {"e2e4+", "e4e5+"}),
               Exception);
  EXPECT_THROW(
      MakeGameStateFromUci(ChessBoard::kStartposFen,
                           {"e2e4+", "e4e5", "a7a6", "g1f3+"}),
      Exception);
}

TEST(GameState, FiltersOpponentKingCaptureDuringSecondHalf) {
  constexpr char kFen[] =
      "k7/8/8/8/8/8/8/1R5K w - - 0 1";
  const auto mid = MakeGameStateFromUci(kFen, {"b1b8+"});
  EXPECT_FALSE(ContainsMove(GenerateDoubleMoveLegalMoves(mid), "b8a8"));
  EXPECT_THROW(MakeGameStateFromUci(kFen, {"b1b8+", "b8a8"}), Exception);
}

TEST(GameState, RejectsAbilityActivationWithoutASecondMove) {
  constexpr char kFen[] =
      "7r/8/7P/8/1n6/8/2k5/K7 w - - 0 1";
  const auto state = MakeGameStateFromUci(kFen, {});
  const MoveList moves = GenerateDoubleMoveLegalMoves(state);
  EXPECT_TRUE(ContainsMove(moves, "h6h7"));
  EXPECT_FALSE(ContainsMove(moves, "h6h7+"));
  EXPECT_THROW(MakeGameStateFromUci(kFen, {"h6h7+"}), Exception);
}

TEST(GameState, PreservesVariantRepetitionsAcrossMidTurnPositions) {
  const std::vector<std::string> burn_both_abilities = {
      "g1f3+", "f3g1", "g8f6+", "f6g8"};
  auto state =
      MakeGameStateFromUci(ChessBoard::kStartposFen, burn_both_abilities);
  EXPECT_EQ(state.CurrentPosition().GetRepetitions(), 0)
      << "Same board with different ability rights is not a repetition";

  const std::vector<std::string> knight_cycle = {
      "g1f3", "g8f6", "f3g1", "f6g8"};
  std::vector<std::string> moves = burn_both_abilities;
  moves.insert(moves.end(), knight_cycle.begin(), knight_cycle.end());
  state = MakeGameStateFromUci(ChessBoard::kStartposFen, moves);
  EXPECT_EQ(state.CurrentPosition().GetRepetitions(), 1);

  moves.insert(moves.end(), knight_cycle.begin(), knight_cycle.end());
  state = MakeGameStateFromUci(ChessBoard::kStartposFen, moves);
  EXPECT_EQ(state.CurrentPosition().GetRepetitions(), 2);
  EXPECT_EQ(state.GetPositionHistory().ComputeGameResult(), GameResult::DRAW);
  EXPECT_TRUE(GenerateDoubleMoveLegalMoves(state).empty());
}

TEST(GameState, MidDoublePawnPushClearsEnPassantAndRoundTrips) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1", "e2e4+"},
      {"4k3/4p3/8/3P4/8/8/8/4K3 b - - 0 1", "e7e5+"},
  };
  for (const auto& [fen, move] : cases) {
    const auto state = MakeGameStateFromUci(fen, {move});
    const Position current = state.CurrentPosition();
    EXPECT_TRUE(current.GetBoard().en_passant().empty());
    const std::string serialized = PositionToFen(current);
    EXPECT_NE(serialized.find(" - 0 "), std::string::npos) << serialized;
    EXPECT_EQ(Position::FromFen(serialized), current);
  }
}

TEST(GameState, FenFullmoveAdvancesOncePerCompletedTurn) {
  const auto ordinary = MakeGameStateFromUci(
      ChessBoard::kStartposFen, {"e2e4", "e7e5"});
  EXPECT_TRUE(PositionToFen(ordinary.CurrentPosition()).ends_with(" 0 2"));

  const auto white_mid =
      MakeGameStateFromUci(ChessBoard::kStartposFen, {"g1f3+"});
  EXPECT_TRUE(PositionToFen(white_mid.CurrentPosition()).ends_with(
      " 1 1 -Bm"));

  const auto white_complete = MakeGameStateFromUci(
      ChessBoard::kStartposFen, {"g1f3+", "f3g1"});
  EXPECT_TRUE(
      PositionToFen(white_complete.CurrentPosition()).ends_with(" 2 1 -B"));

  const auto both_complete = MakeGameStateFromUci(
      ChessBoard::kStartposFen,
      {"g1f3+", "f3g1", "g8f6+", "f6g8"});
  EXPECT_TRUE(
      PositionToFen(both_complete.CurrentPosition()).ends_with(" 4 2 --"));
}

TEST(GameState, RejectsInconsistentOrExtraAbilityFenState) {
  EXPECT_THROW(
      Position::FromFen("4k3/8/8/8/8/8/8/4K3 w - - 0 1 WBm"),
      Exception);
  EXPECT_THROW(
      Position::FromFen("4k3/8/8/8/8/8/8/4K3 b - - 0 1 WBm"),
      Exception);
  EXPECT_THROW(
      Position::FromFen("4k3/8/8/8/8/8/8/4K3 w - - 0 1 -Bm extra"),
      Exception);
}

TEST(GameState, GeneratesPromotionAbilityTokens) {
  const auto state = MakeGameStateFromUci(
      "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", {});
  const MoveList moves = GenerateDoubleMoveLegalMoves(state);
  for (const std::string_view suffix : {"n", "b", "r", "q"}) {
    EXPECT_TRUE(ContainsMove(moves, std::string("a7a8") +
                                       std::string(suffix)));
    EXPECT_TRUE(ContainsMove(moves, std::string("a7a8") +
                                       std::string(suffix) + "+"));
  }
}

TEST(GameState, TerminalPositionsHaveNoLegalMovesButMidTurnIsOngoing) {
  const auto checkmate = MakeGameStateFromUci(
      "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1 --", {});
  EXPECT_EQ(checkmate.GetPositionHistory().ComputeGameResult(),
            GameResult::WHITE_WON);
  EXPECT_TRUE(GenerateDoubleMoveLegalMoves(checkmate).empty());

  const auto mid = MakeGameStateFromUci(
      "4k3/8/8/8/8/8/8/4K3 w - - 100 1 -Bm", {});
  EXPECT_EQ(mid.GetPositionHistory().ComputeGameResult(),
            GameResult::UNDECIDED);
  EXPECT_FALSE(GenerateDoubleMoveLegalMoves(mid).empty());
}

TEST(UciOpeningSchedule, ParsesColorAndDoubleMove) {
  const auto opening = ParseUciOpeningScheduleLine(
      "dm-001\tblack\te2e4+ g1f3 e7e5", 7);

  ASSERT_TRUE(opening.id);
  EXPECT_EQ(*opening.id, "dm-001");
  ASSERT_TRUE(opening.player1_black);
  EXPECT_TRUE(*opening.player1_black);
  ASSERT_TRUE(opening.uci_moves);
  EXPECT_EQ(*opening.uci_moves,
            (std::vector<std::string>{"e2e4+", "g1f3", "e7e5"}));
}

TEST(UciOpeningSchedule, RejectsMidDoubleEndpoint) {
  EXPECT_THROW(
      ParseUciOpeningScheduleLine("dm-002\twhite\te2e4+", 2), Exception);
}

TEST(Position, SetFenGetFen) {
  std::vector<Position> positions;
  ChessBoard board;
  std::vector<std::string> source_fens = {
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 1 1",
      // has en_passant space e3 - black to move
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq e3 1 1",
      // has en_passant space c6 - white to move
      "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 1 1",
      "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
      "3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30",
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 "
      "10",
      "8/8/8/4k3/8/8/2K5/8 w - - 0 1", "8/8/8/4k3/1N6/8/2K5/8 w - - 0 1"};
  for (size_t i = 0; i < source_fens.size(); i++) {
    board.Clear();
    PositionHistory history;
    int no_capture_ply;
    int game_move;
    board.SetFromFen(source_fens[i], &no_capture_ply, &game_move);
    history.Reset(board, no_capture_ply,
                  2 * game_move - (board.flipped() ? 1 : 2));
    Position pos = history.Last();
    std::string target_fen = PositionToFen(pos);
    EXPECT_EQ(source_fens[i], target_fen);
  }
}

// https://github.com/LeelaChessZero/lc0/issues/209
TEST(PositionHistory, ComputeLastMoveRepetitionsWithoutLegalEnPassant) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("f7f8"));
  history.Append(history.Last().GetBoard().ParseMove("f2f4"));
  history.Append(history.Last().GetBoard().ParseMove("d7h7"));
  history.Append(history.Last().GetBoard().ParseMove("c4d3"));
  history.Append(history.Last().GetBoard().ParseMove("h7d7"));
  history.Append(history.Last().GetBoard().ParseMove("d3c4"));
  int history_idx = history.GetLength() - 1;
  const Position& repeated_position = history.GetPositionAt(history_idx);
  EXPECT_EQ(repeated_position.GetRepetitions(), 1);
}

TEST(PositionHistory, ComputeLastMoveRepetitionsWithLegalEnPassant) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP2p1/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("f7f8"));
  history.Append(history.Last().GetBoard().ParseMove("f2f4"));
  history.Append(history.Last().GetBoard().ParseMove("d7h7"));
  history.Append(history.Last().GetBoard().ParseMove("c4d3"));
  history.Append(history.Last().GetBoard().ParseMove("h7d7"));
  history.Append(history.Last().GetBoard().ParseMove("d3c4"));
  int history_idx = history.GetLength() - 1;
  const Position& repeated_position = history.GetPositionAt(history_idx);
  EXPECT_EQ(repeated_position.GetRepetitions(), 0);
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveCurent) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("f7f8"));
  history.Append(history.Last().GetBoard().ParseMove("f2f4"));
  history.Append(history.Last().GetBoard().ParseMove("d7h7"));
  history.Append(history.Last().GetBoard().ParseMove("c4d3"));
  history.Append(history.Last().GetBoard().ParseMove("h7d7"));
  history.Append(history.Last().GetBoard().ParseMove("d3c4"));
  EXPECT_TRUE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveBefore) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("f7f8"));
  history.Append(history.Last().GetBoard().ParseMove("f2f4"));
  history.Append(history.Last().GetBoard().ParseMove("d7h7"));
  history.Append(history.Last().GetBoard().ParseMove("c4d3"));
  history.Append(history.Last().GetBoard().ParseMove("h7d7"));
  history.Append(history.Last().GetBoard().ParseMove("d3c4"));
  history.Append(history.Last().GetBoard().ParseMove("d7e7"));
  EXPECT_TRUE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveOlder) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("f7f8"));
  history.Append(history.Last().GetBoard().ParseMove("f2f4"));
  history.Append(history.Last().GetBoard().ParseMove("d7h7"));
  history.Append(history.Last().GetBoard().ParseMove("c4d3"));
  history.Append(history.Last().GetBoard().ParseMove("h7d7"));
  history.Append(history.Last().GetBoard().ParseMove("d3c4"));
  history.Append(history.Last().GetBoard().ParseMove("d7e7"));
  history.Append(history.Last().GetBoard().ParseMove("c4b4"));
  EXPECT_TRUE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveBeforeZero) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("f7f8"));
  history.Append(history.Last().GetBoard().ParseMove("f2f4"));
  history.Append(history.Last().GetBoard().ParseMove("d7h7"));
  history.Append(history.Last().GetBoard().ParseMove("c4d3"));
  history.Append(history.Last().GetBoard().ParseMove("h7d7"));
  history.Append(history.Last().GetBoard().ParseMove("d3c4"));
  history.Append(history.Last().GetBoard().ParseMove("d7e7"));
  history.Append(history.Last().GetBoard().ParseMove("c4b4"));
  history.Append(history.Last().GetBoard().ParseMove("h5h4"));
  EXPECT_FALSE(history.DidRepeatSinceLastZeroingMove());
}

TEST(PositionHistory, DidRepeatSinceLastZeroingMoveNeverRepeated) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3b4/rp1r1k2/8/1RP2p1p/p1KP4/P3P2P/5P2/1R2B3 b - - 2 30");
  history.Reset(board, 2, 30);
  history.Append(history.Last().GetBoard().ParseMove("f7f8"));
  history.Append(history.Last().GetBoard().ParseMove("f2f4"));
  EXPECT_FALSE(history.DidRepeatSinceLastZeroingMove());
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
