#include "neural/dual_backend.h"

#include <algorithm>
#include <cassert>

namespace lczero {

DualBackend::DualBackend(std::shared_ptr<Backend> variant_backend,
                         std::shared_ptr<Backend> standard_backend)
    : variant_backend_(std::move(variant_backend)),
      standard_backend_(std::move(standard_backend)) {}

BackendAttributes DualBackend::GetAttributes() const {
  // Use variant backend's attributes as the primary attributes.
  return variant_backend_->GetAttributes();
}

namespace {

// Returns true if the position has exhausted both players' double-move
// abilities, meaning it is a standard chess position.
bool IsStandardChessPosition(const EvalPosition& pos) {
  if (pos.pos.empty()) return false;
  const Position& last = pos.pos.back();
  return !last.OurDoublemoveAvailable() && !last.TheirDoublemoveAvailable();
}

// A BackendComputation that routes AddInput calls to either the variant or
// standard network based on the position's ability state.
class DualBackendComputation : public BackendComputation {
 public:
  DualBackendComputation(Backend* variant_backend, Backend* standard_backend)
      : variant_comp_(variant_backend->CreateComputation()),
        standard_comp_(standard_backend->CreateComputation()) {}

  size_t UsedBatchSize() const override {
    return variant_comp_->UsedBatchSize() + standard_comp_->UsedBatchSize();
  }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    if (IsStandardChessPosition(pos)) {
      return standard_comp_->AddInput(pos, result);
    } else {
      return variant_comp_->AddInput(pos, result);
    }
  }

  void ComputeBlocking() override {
    // Dispatch both computations. Order doesn't matter — they are independent.
    if (variant_comp_->UsedBatchSize() > 0) {
      variant_comp_->ComputeBlocking();
    }
    if (standard_comp_->UsedBatchSize() > 0) {
      standard_comp_->ComputeBlocking();
    }
  }

 private:
  std::unique_ptr<BackendComputation> variant_comp_;
  std::unique_ptr<BackendComputation> standard_comp_;
};

}  // namespace

std::unique_ptr<BackendComputation> DualBackend::CreateComputation() {
  return std::make_unique<DualBackendComputation>(variant_backend_.get(),
                                                  standard_backend_.get());
}

std::optional<EvalResult> DualBackend::GetCachedEvaluation(
    const EvalPosition& pos) {
  if (IsStandardChessPosition(pos)) {
    return standard_backend_->GetCachedEvaluation(pos);
  }
  return variant_backend_->GetCachedEvaluation(pos);
}

}  // namespace lczero
