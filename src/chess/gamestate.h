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

#pragma once

#include <string>
#include <vector>

#include "chess/position.h"

namespace lczero {

// A structure that is passed to Search/SearchEnvironment to provide the game
// state. Somewhat mirrors usi `position <fen> moves ...` command.
struct GameState {
  Position startpos;
  std::vector<Move> moves;

  // Reconstructs the complete position history, including repetition counts.
  PositionHistory GetPositionHistory() const;
  // Returns the position of the last move in the list.
  Position CurrentPosition() const;
  // Returns positions after all moves, including the starting and the last one.
  std::vector<Position> GetPositions() const;
};

// Returns ordinary moves allowed from a position under the phase-specific
// double-move rules. In particular, opponent-king captures are excluded from
// the second half.
MoveList GenerateDoubleMoveBoardMoves(const Position& position);

// Returns whether a legal ordinary move can begin a completable double turn.
// The caller must supply a move from GenerateDoubleMoveBoardMoves(position).
bool HasLegalDoubleMoveContinuation(const Position& position,
                                    Move first_move);

// Returns the moves that are legal under the double-move variant rules.
//
// Moves are in the side-to-move-relative representation used by ChessBoard.
// Ability activations carry Move::is_ability_activation(). Callers emitting
// UCI must flip moves when CurrentPosition().IsBlackToMove().
MoveList GenerateDoubleMoveLegalMoves(const GameState& state);

// Parses the same coordinate-move sequence accepted by the UCI
// `position <fen> moves ...` command, including double-move activation `+`,
// and validates every move against the variant rules.
GameState MakeGameStateFromUci(const std::string& fen,
                               const std::vector<std::string>& moves);

}  // namespace lczero
