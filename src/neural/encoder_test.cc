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

#include "src/neural/encoder.h"

#include <gtest/gtest.h>

#include <string>

namespace lczero {

auto kAllSquaresMask = std::numeric_limits<std::uint64_t>::max();

TEST(EncodePositionForNN, EncodeStartPosition) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  InputPlanes encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::NO, nullptr);

  InputPlane our_pawns_plane = encoded_planes[0];
  auto our_pawns_mask = 0ull;
  for (auto i = 0; i < 8; i++) {
    // First pawn is at square a2 (position 8)
    // Last pawn is at square h2 (position 8 + 7 = 15)
    our_pawns_mask |= 1ull << (8 + i);
  }
  EXPECT_EQ(our_pawns_plane.mask, our_pawns_mask);
  EXPECT_EQ(our_pawns_plane.value, 1.0f);

  InputPlane our_knights_plane = encoded_planes[1];
  EXPECT_EQ(our_knights_plane.mask, (1ull << 1) | (1ull << 6));
  EXPECT_EQ(our_knights_plane.value, 1.0f);

  InputPlane our_bishops_plane = encoded_planes[2];
  EXPECT_EQ(our_bishops_plane.mask, (1ull << 2) | (1ull << 5));
  EXPECT_EQ(our_bishops_plane.value, 1.0f);

  InputPlane our_rooks_plane = encoded_planes[3];
  EXPECT_EQ(our_rooks_plane.mask, 1ull | (1ull << 7));
  EXPECT_EQ(our_rooks_plane.value, 1.0f);

  InputPlane our_queens_plane = encoded_planes[4];
  EXPECT_EQ(our_queens_plane.mask, 1ull << 3);
  EXPECT_EQ(our_queens_plane.value, 1.0f);

  InputPlane our_king_plane = encoded_planes[5];
  EXPECT_EQ(our_king_plane.mask, 1ull << 4);
  EXPECT_EQ(our_king_plane.value, 1.0f);

  // Sanity check opponent's pieces
  InputPlane their_king_plane = encoded_planes[11];
  auto their_king_row = 7;
  auto their_king_col = 4;
  EXPECT_EQ(their_king_plane.mask,
            1ull << (8 * their_king_row + their_king_col));
  EXPECT_EQ(their_king_plane.value, 1.0f);

  // Start of game, no history.
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 13; j++) {
      InputPlane zeroed_history = encoded_planes[13 + i * 13 + j];
      EXPECT_EQ(zeroed_history.mask, 0ull);
    }
  }

  // Auxiliary planes

  // It's the start of the game, so all castlings should be allowed.
  for (auto i = 0; i < 4; i++) {
    InputPlane can_castle_plane = encoded_planes[13 * 8 + i];
    EXPECT_EQ(can_castle_plane.mask, kAllSquaresMask);
    EXPECT_EQ(can_castle_plane.value, 1.0f);
  }

  InputPlane we_are_black_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(we_are_black_plane.mask, 0ull);

  InputPlane fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 0.0f);

  // We no longer encode the move count, so that plane should be all zeros
  InputPlane zeroed_move_count_plane = encoded_planes[13 * 8 + 6];
  EXPECT_EQ(zeroed_move_count_plane.mask, 0ull);

  InputPlane all_ones_plane = encoded_planes[13 * 8 + 7];
  EXPECT_EQ(all_ones_plane.mask, kAllSquaresMask);
  EXPECT_EQ(all_ones_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeStartPositionFormat2) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  InputPlanes encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE, history, 8,
      FillEmptyHistory::NO, nullptr);

  InputPlane our_pawns_plane = encoded_planes[0];
  auto our_pawns_mask = 0ull;
  for (auto i = 0; i < 8; i++) {
    // First pawn is at square a2 (position 8)
    // Last pawn is at square h2 (position 8 + 7 = 15)
    our_pawns_mask |= 1ull << (8 + i);
  }
  EXPECT_EQ(our_pawns_plane.mask, our_pawns_mask);
  EXPECT_EQ(our_pawns_plane.value, 1.0f);

  InputPlane our_knights_plane = encoded_planes[1];
  EXPECT_EQ(our_knights_plane.mask, (1ull << 1) | (1ull << 6));
  EXPECT_EQ(our_knights_plane.value, 1.0f);

  InputPlane our_bishops_plane = encoded_planes[2];
  EXPECT_EQ(our_bishops_plane.mask, (1ull << 2) | (1ull << 5));
  EXPECT_EQ(our_bishops_plane.value, 1.0f);

  InputPlane our_rooks_plane = encoded_planes[3];
  EXPECT_EQ(our_rooks_plane.mask, 1ull | (1ull << 7));
  EXPECT_EQ(our_rooks_plane.value, 1.0f);

  InputPlane our_queens_plane = encoded_planes[4];
  EXPECT_EQ(our_queens_plane.mask, 1ull << 3);
  EXPECT_EQ(our_queens_plane.value, 1.0f);

  InputPlane our_king_plane = encoded_planes[5];
  EXPECT_EQ(our_king_plane.mask, 1ull << 4);
  EXPECT_EQ(our_king_plane.value, 1.0f);

  // Sanity check opponent's pieces
  InputPlane their_king_plane = encoded_planes[11];
  auto their_king_row = 7;
  auto their_king_col = 4;
  EXPECT_EQ(their_king_plane.mask,
            1ull << (8 * their_king_row + their_king_col));
  EXPECT_EQ(their_king_plane.value, 1.0f);

  // Start of game, no history.
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 13; j++) {
      InputPlane zeroed_history = encoded_planes[13 + i * 13 + j];
      EXPECT_EQ(zeroed_history.mask, 0ull);
    }
  }

  // Auxiliary planes

  // Queen side castling at game start.
  InputPlane can_castle_plane = encoded_planes[13 * 8 + 0];
  EXPECT_EQ(can_castle_plane.mask, 1ull | (1ull << 56));
  EXPECT_EQ(can_castle_plane.value, 1.0f);
  // king side castling at game start.
  can_castle_plane = encoded_planes[13 * 8 + 1];
  EXPECT_EQ(can_castle_plane.mask, 1ull << 7 | (1ull << 63));
  EXPECT_EQ(can_castle_plane.value, 1.0f);

  // Zeroed castling planes.
  InputPlane zeroed_castling_plane = encoded_planes[13 * 8 + 2];
  EXPECT_EQ(zeroed_castling_plane.mask, 0ull);
  zeroed_castling_plane = encoded_planes[13 * 8 + 3];
  EXPECT_EQ(zeroed_castling_plane.mask, 0ull);

  InputPlane we_are_black_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(we_are_black_plane.mask, 0ull);

  InputPlane fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 0.0f);

  // We no longer encode the move count, so that plane should be all zeros
  InputPlane zeroed_move_count_plane = encoded_planes[13 * 8 + 6];
  EXPECT_EQ(zeroed_move_count_plane.mask, 0ull);

  InputPlane all_ones_plane = encoded_planes[13 * 8 + 7];
  EXPECT_EQ(all_ones_plane.mask, kAllSquaresMask);
  EXPECT_EQ(all_ones_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeStartPositionFormat3) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  InputPlanes encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, nullptr);

  InputPlane our_pawns_plane = encoded_planes[0];
  auto our_pawns_mask = 0ull;
  for (auto i = 0; i < 8; i++) {
    // First pawn is at square a2 (position 8)
    // Last pawn is at square h2 (position 8 + 7 = 15)
    our_pawns_mask |= 1ull << (8 + i);
  }
  EXPECT_EQ(our_pawns_plane.mask, our_pawns_mask);
  EXPECT_EQ(our_pawns_plane.value, 1.0f);

  InputPlane our_knights_plane = encoded_planes[1];
  EXPECT_EQ(our_knights_plane.mask, (1ull << 1) | (1ull << 6));
  EXPECT_EQ(our_knights_plane.value, 1.0f);

  InputPlane our_bishops_plane = encoded_planes[2];
  EXPECT_EQ(our_bishops_plane.mask, (1ull << 2) | (1ull << 5));
  EXPECT_EQ(our_bishops_plane.value, 1.0f);

  InputPlane our_rooks_plane = encoded_planes[3];
  EXPECT_EQ(our_rooks_plane.mask, 1ull | (1ull << 7));
  EXPECT_EQ(our_rooks_plane.value, 1.0f);

  InputPlane our_queens_plane = encoded_planes[4];
  EXPECT_EQ(our_queens_plane.mask, 1ull << 3);
  EXPECT_EQ(our_queens_plane.value, 1.0f);

  InputPlane our_king_plane = encoded_planes[5];
  EXPECT_EQ(our_king_plane.mask, 1ull << 4);
  EXPECT_EQ(our_king_plane.value, 1.0f);

  // Sanity check opponent's pieces
  InputPlane their_king_plane = encoded_planes[11];
  auto their_king_row = 7;
  auto their_king_col = 4;
  EXPECT_EQ(their_king_plane.mask,
            1ull << (8 * their_king_row + their_king_col));
  EXPECT_EQ(their_king_plane.value, 1.0f);

  // Start of game, no history.
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 13; j++) {
      InputPlane zeroed_history = encoded_planes[13 + i * 13 + j];
      EXPECT_EQ(zeroed_history.mask, 0ull);
    }
  }

  // Auxiliary planes

  // Queen side castling at game start.
  InputPlane can_castle_plane = encoded_planes[13 * 8 + 0];
  EXPECT_EQ(can_castle_plane.mask, 1ull | (1ull << 56));
  EXPECT_EQ(can_castle_plane.value, 1.0f);
  // king side castling at game start.
  can_castle_plane = encoded_planes[13 * 8 + 1];
  EXPECT_EQ(can_castle_plane.mask, 1ull << 7 | (1ull << 63));
  EXPECT_EQ(can_castle_plane.value, 1.0f);

  // Zeroed castling planes.
  InputPlane zeroed_castling_plane = encoded_planes[13 * 8 + 2];
  EXPECT_EQ(zeroed_castling_plane.mask, 0ull);
  zeroed_castling_plane = encoded_planes[13 * 8 + 3];
  EXPECT_EQ(zeroed_castling_plane.mask, 0ull);

  InputPlane enpassant_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(enpassant_plane.mask, 0ull);

  InputPlane fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 0.0f);

  // We no longer encode the move count, so that plane should be all zeros
  InputPlane zeroed_move_count_plane = encoded_planes[13 * 8 + 6];
  EXPECT_EQ(zeroed_move_count_plane.mask, 0ull);

  InputPlane all_ones_plane = encoded_planes[13 * 8 + 7];
  EXPECT_EQ(all_ones_plane.mask, kAllSquaresMask);
  EXPECT_EQ(all_ones_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeFiftyMoveCounter) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  // 1. Nf3
  history.Append(history.Last().GetBoard().ParseMove("g1f3"));

  InputPlanes encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::NO, nullptr);

  InputPlane we_are_black_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(we_are_black_plane.mask, kAllSquaresMask);
  EXPECT_EQ(we_are_black_plane.value, 1.0f);

  InputPlane fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 1.0f);

  // 1. Nf3 Nf6
  history.Append(history.Last().GetBoard().ParseMove("g8f6"));

  encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::NO, nullptr);

  we_are_black_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(we_are_black_plane.mask, 0ull);

  fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 2.0f);
}

TEST(EncodePositionForNN, EncodeFiftyMoveCounterFormat3) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);

  // 1. Nf3
  history.Append(history.Last().GetBoard().ParseMove("g1f3"));

  InputPlanes encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, nullptr);

  InputPlane enpassant_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(enpassant_plane.mask, 0ull);

  InputPlane fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 1.0f);

  // 1. Nf3 Nf6
  history.Append(history.Last().GetBoard().ParseMove("g8f6"));

  encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, nullptr);

  enpassant_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(enpassant_plane.mask, 0ull);

  fifty_move_counter_plane = encoded_planes[13 * 8 + 5];
  EXPECT_EQ(fifty_move_counter_plane.mask, kAllSquaresMask);
  EXPECT_EQ(fifty_move_counter_plane.value, 2.0f);
}

TEST(EncodePositionForNN, EncodeEndGameFormat1) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3r4/4k3/8/1K6/8/8/8/8 w - - 0 1");
  history.Reset(board, 0, 1);

  int transform;
  InputPlanes encoded_planes =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::NO, &transform);

  EXPECT_EQ(transform, NoTransform);

  InputPlane our_king_plane = encoded_planes[5];
  EXPECT_EQ(our_king_plane.mask, 1ull << 33);
  EXPECT_EQ(our_king_plane.value, 1.0f);
  InputPlane their_king_plane = encoded_planes[11];
  EXPECT_EQ(their_king_plane.mask, 1ull << 52);
  EXPECT_EQ(their_king_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeEndGameFormat3) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3r4/4k3/8/1K6/8/8/8/8 w - - 0 1");
  history.Reset(board, 0, 1);

  int transform;
  InputPlanes encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, &transform);

  EXPECT_EQ(transform, FlipTransform | MirrorTransform | TransposeTransform);

  InputPlane our_king_plane = encoded_planes[5];
  EXPECT_EQ(our_king_plane.mask, 1ull << 12);
  EXPECT_EQ(our_king_plane.value, 1.0f);
  InputPlane their_king_plane = encoded_planes[11];
  EXPECT_EQ(their_king_plane.mask, 1ull << 38);
  EXPECT_EQ(their_king_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeEndGameKingOnDiagonalFormat3) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("3r4/4k3/2K5/8/8/8/8/8 w - - 0 1");
  history.Reset(board, 0, 1);

  int transform;
  InputPlanes encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, &transform);

  // After mirroring transforms, our king is on diagonal and other pieces are
  // all below the diagonal, so transposing will increase the value of ours |
  // theirs.
  EXPECT_EQ(transform, FlipTransform | MirrorTransform);

  InputPlane our_king_plane = encoded_planes[5];
  EXPECT_EQ(our_king_plane.mask, 1ull << 21);
  EXPECT_EQ(our_king_plane.value, 1.0f);
  InputPlane their_king_plane = encoded_planes[11];
  EXPECT_EQ(their_king_plane.mask, 1ull << 11);
  EXPECT_EQ(their_king_plane.value, 1.0f);
}

TEST(EncodePositionForNN, EncodeEnpassantFormat3) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);
  // Move to en passant.
  history.Append(history.Last().GetBoard().ParseMove("e2e4"));
  history.Append(history.Last().GetBoard().ParseMove("g7g6"));
  history.Append(history.Last().GetBoard().ParseMove("e4e5"));
  history.Append(history.Last().GetBoard().ParseMove("f7f5"));

  InputPlanes encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, nullptr);

  InputPlane enpassant_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(enpassant_plane.mask, 1ull << 61);

  // Pawn move, no history.
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 13; j++) {
      InputPlane zeroed_history = encoded_planes[13 + i * 13 + j];
      EXPECT_EQ(zeroed_history.mask, 0ull);
    }
  }

  // Boring move.
  history.Append(history.Last().GetBoard().ParseMove("g1f3"));

  encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, nullptr);

  // No more en passant bit.
  enpassant_plane = encoded_planes[13 * 8 + 4];
  EXPECT_EQ(enpassant_plane.mask, 0ull);

  // Previous was en passant, no history.
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 13; j++) {
      InputPlane zeroed_history = encoded_planes[13 + i * 13 + j];
      EXPECT_EQ(zeroed_history.mask, 0ull);
    }
  }

  // Another boring move.
  history.Append(history.Last().GetBoard().ParseMove("g8f5"));

  encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, nullptr);

  // Should be one plane of history.
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 13; j++) {
      InputPlane zeroed_history = encoded_planes[13 + i * 13 + j];
      // 13th plane of first layer is repeats and there are none, so it should
      // be empty.
      if (i == 0 && j < 12) {
        EXPECT_NE(zeroed_history.mask, 0ull);
      } else {
        EXPECT_EQ(zeroed_history.mask, 0ull);
      }
    }
  }
}

TEST(EncodePositionForNN, EncodeEarlyGameFlipFormat3) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);
  // Move to break castling and king offside.
  history.Append(history.Last().GetBoard().ParseMove("e2e4"));
  history.Append(history.Last().GetBoard().ParseMove("e7e5"));
  history.Append(history.Last().GetBoard().ParseMove("e1e2"));
  history.Append(history.Last().GetBoard().ParseMove("e8e7"));
  history.Append(history.Last().GetBoard().ParseMove("e2d3"));
  // Their king offside, but not ours.

  int transform;
  InputPlanes encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, &transform);

  EXPECT_EQ(transform, NoTransform);

  InputPlane our_king_plane = encoded_planes[5];
  EXPECT_EQ(our_king_plane.mask, 1ull << 12);
  EXPECT_EQ(our_king_plane.value, 1.0f);
  InputPlane their_king_plane = encoded_planes[11];
  EXPECT_EQ(their_king_plane.mask, 1ull << 43);
  EXPECT_EQ(their_king_plane.value, 1.0f);

  history.Append(history.Last().GetBoard().ParseMove("e7e6"));

  // Our king offside, but theirs is not.
  encoded_planes = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION, history, 8,
      FillEmptyHistory::NO, &transform);

  EXPECT_EQ(transform, FlipTransform);

  our_king_plane = encoded_planes[5];
  EXPECT_EQ(our_king_plane.mask, 1ull << 20);
  EXPECT_EQ(our_king_plane.value, 1.0f);
  their_king_plane = encoded_planes[11];
  EXPECT_EQ(their_king_plane.mask, 1ull << 43);
  EXPECT_EQ(their_king_plane.value, 1.0f);
}

TEST(EncodePositionForNN, RepeatCurrentIgnoresRealHistory) {
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen(ChessBoard::kStartposFen);
  history.Reset(board, 0, 1);
  history.Append(history.Last().GetBoard().ParseMove("e2e4"));
  history.Append(history.Last().GetBoard().ParseMove("e7e5"));
  history.Append(history.Last().GetBoard().ParseMove("g1f3"));

  InputPlanes repeated =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::REPEAT_CURRENT,
                          nullptr);

  for (int slot = 1; slot < 8; ++slot) {
    for (int plane = 0; plane < kPlanesPerBoard; ++plane) {
      SCOPED_TRACE("slot=" + std::to_string(slot) +
                   " plane=" + std::to_string(plane));
      EXPECT_EQ(repeated[slot * kPlanesPerBoard + plane].mask,
                repeated[plane].mask);
      EXPECT_EQ(repeated[slot * kPlanesPerBoard + plane].value,
                repeated[plane].value);
    }
  }

  InputPlanes normal_history =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::ALWAYS, nullptr);
  bool slot_one_differs = false;
  for (int plane = 0; plane < kPlanesPerBoard; ++plane) {
    slot_one_differs |=
        normal_history[kPlanesPerBoard + plane].mask != normal_history[plane].mask;
  }
  EXPECT_TRUE(slot_one_differs);
}

TEST(EncodePositionForNN, RepeatCurrentReconstructsEnPassant) {
  // Networks perceive en passant via the i=0-vs-i=1 history delta (the
  // pushing pawn was on rank 2/7 one ply ago, now on rank 4/5).  Under the
  // old `repeat_current` semantics every history slot copied the current
  // board, erasing that delta and making the network blind to e.p.
  //
  // Set up a real attackable e.p. position via FEN+move: black pawn on c4,
  // then white plays b2b4 — this is the canonical user-reported case.
  // After the move, side-to-move is black, the e.p. flag is set on b3, and
  // black's c4 pawn can take b3 en passant.
  ChessBoard board;
  PositionHistory history;
  board.SetFromFen("4k3/8/8/8/2p5/8/1P6/4K3 w - - 0 1");
  history.Reset(board, 0, 1);
  history.Append(history.Last().GetBoard().ParseMove("b2b4"));
  // Sanity: the engine should have set the e.p. flag (because c4 attacks b3).
  ASSERT_FALSE(history.Last().GetBoard().en_passant().empty())
      << "test setup is broken: no e.p. attacker present";

  InputPlanes repeated =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          history, 8, FillEmptyHistory::REPEAT_CURRENT,
                          nullptr);

  // Verify slot 1+ shows the pushing pawn ghosted back to its pre-push
  // square.  We check both "ours" (plane 0) and "theirs" (plane 6) pawn
  // planes, because lc0 mirrors the e.p. flag to whichever side ends up
  // representing the pusher in side-to-move POV.
  const uint64_t cur_ours = repeated[0].mask;
  const uint64_t cur_theirs = repeated[6].mask;
  for (int slot = 1; slot < 8; ++slot) {
    const uint64_t slot_ours = repeated[slot * kPlanesPerBoard + 0].mask;
    const uint64_t slot_theirs = repeated[slot * kPlanesPerBoard + 6].mask;
    SCOPED_TRACE("slot=" + std::to_string(slot));
    EXPECT_TRUE(slot_ours != cur_ours || slot_theirs != cur_theirs)
        << "expected reconstructed pre-push pawn position in slot " << slot
        << " (slot 0 ours=0x" << std::hex << cur_ours << " theirs=0x"
        << cur_theirs << ", slot " << std::dec << slot << " ours=0x"
        << std::hex << slot_ours << " theirs=0x" << slot_theirs << std::dec
        << ")";
  }

  // All reconstructed slots agree with each other (slot 1 == slot 7 etc.).
  for (int slot = 2; slot < 8; ++slot) {
    SCOPED_TRACE("slot=" + std::to_string(slot));
    for (int plane = 0; plane < kPlanesPerBoard; ++plane) {
      EXPECT_EQ(repeated[slot * kPlanesPerBoard + plane].mask,
                repeated[kPlanesPerBoard + plane].mask);
    }
  }

  // Cross-check: when the position is loaded purely from FEN with no real
  // history, `fen_only` already invokes the same ghost-pawn trick on every
  // i>0 slot, so repeat_current must produce identical planes in that case.
  ChessBoard fen_board;
  fen_board.SetFromFen("4k3/8/8/8/1Pp5/8/8/4K3 b - b3 0 1");
  PositionHistory fen_history;
  fen_history.Reset(fen_board, 0, 1);
  ASSERT_FALSE(fen_history.Last().GetBoard().en_passant().empty())
      << "FEN setup is broken: no e.p. flag";

  InputPlanes fen_repeat =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          fen_history, 8, FillEmptyHistory::REPEAT_CURRENT,
                          nullptr);
  InputPlanes fen_only =
      EncodePositionForNN(pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
                          fen_history, 8, FillEmptyHistory::FEN_ONLY, nullptr);
  for (int slot = 0; slot < 8; ++slot) {
    for (int plane = 0; plane < kPlanesPerBoard; ++plane) {
      SCOPED_TRACE("slot=" + std::to_string(slot) +
                   " plane=" + std::to_string(plane));
      EXPECT_EQ(fen_repeat[slot * kPlanesPerBoard + plane].mask,
                fen_only[slot * kPlanesPerBoard + plane].mask);
    }
  }
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}
