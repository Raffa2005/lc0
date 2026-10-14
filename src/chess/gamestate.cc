/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "chess/gamestate.h"

#include <algorithm>

#include "utils/exception.h"

namespace lczero {
namespace {

MoveList GenerateDoubleMoveLegalMoves(const PositionHistory& history) {
  if (history.ComputeGameResult() != GameResult::UNDECIDED) return {};

  const Position& position = history.Last();
  MoveList legal_moves = GenerateDoubleMoveBoardMoves(position);

  if (position.OurDoublemoveAvailable() &&
      !position.IsMidDoublemove()) {
    const size_t ordinary_move_count = legal_moves.size();
    legal_moves.reserve(ordinary_move_count * 2);
    for (size_t i = 0; i < ordinary_move_count; ++i) {
      if (HasLegalDoubleMoveContinuation(position, legal_moves[i])) {
        legal_moves.push_back(legal_moves[i].WithAbilityActivation());
      }
    }
  }
  return legal_moves;
}

void AppendMove(PositionHistory* history, Move move) {
  if (move.is_ability_activation()) {
    history->Append(move.WithoutAbilityActivation(),
                    /*is_first_of_double=*/true);
  } else {
    history->Append(move);
  }
}

}  // namespace

MoveList GenerateDoubleMoveBoardMoves(const Position& position) {
  const ChessBoard& board = position.GetBoard();
  MoveList legal_moves = board.GenerateLegalMoves();

  // The first half may leave the opponent king attacked. Capturing it with the
  // second half is not a move in chess, so remove that pseudo-legal artifact.
  if (position.IsMidDoublemove()) {
    const Square their_king = board.their_king();
    legal_moves.erase(
        std::remove_if(legal_moves.begin(), legal_moves.end(),
                       [their_king](const Move& move) {
                         return move.to() == their_king;
                       }),
        legal_moves.end());
  }
  return legal_moves;
}

bool HasLegalDoubleMoveContinuation(const Position& position,
                                    Move first_move) {
  if (position.IsMidDoublemove() ||
      !position.OurDoublemoveAvailable()) {
    return false;
  }
  const Position mid_position(position,
                              first_move.WithoutAbilityActivation(),
                              /*is_first_of_double=*/true);
  return !GenerateDoubleMoveBoardMoves(mid_position).empty();
}

GameState MakeGameStateFromUci(const std::string& fen,
                               const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(fen);
  PositionHistory history;
  history.Reset(state.startpos);
  state.moves.reserve(moves.size());

  for (size_t index = 0; index < moves.size(); ++index) {
    const std::string& move_string = moves[index];
    const Position& position = history.Last();
    Move move;
    try {
      move = position.GetBoard().ParseMove(move_string);
    } catch (const Exception& exception) {
      throw Exception("Invalid position move " + std::to_string(index + 1) +
                      " (" + move_string + "): " + exception.what());
    }

    const MoveList legal_moves = GenerateDoubleMoveLegalMoves(history);
    if (std::find(legal_moves.begin(), legal_moves.end(), move) ==
        legal_moves.end()) {
      std::string reason = "move is not legal";
      if (move.is_ability_activation()) {
        if (position.IsMidDoublemove()) {
          reason = "cannot activate an ability during a double move";
        } else if (!position.OurDoublemoveAvailable()) {
          reason = "double-move ability is not available";
        }
      }
      throw Exception("Invalid position move " + std::to_string(index + 1) +
                      " (" + move_string + "): " + reason);
    }

    state.moves.push_back(move);
    AppendMove(&history, move);
  }
  return state;
}

PositionHistory GameState::GetPositionHistory() const {
  PositionHistory history;
  history.Reset(startpos);
  history.Reserve(moves.size() + 1);
  for (Move move : moves) AppendMove(&history, move);
  return history;
}

Position GameState::CurrentPosition() const {
  return GetPositionHistory().Last();
}

std::vector<Position> GameState::GetPositions() const {
  const PositionHistory history = GetPositionHistory();
  const std::span<const Position> positions = history.GetPositions();
  return {positions.begin(), positions.end()};
}

MoveList GenerateDoubleMoveLegalMoves(const GameState& state) {
  return GenerateDoubleMoveLegalMoves(state.GetPositionHistory());
}

}  // namespace lczero
