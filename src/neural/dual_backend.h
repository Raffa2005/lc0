#pragma once

#include <memory>

#include "neural/backend.h"

namespace lczero {

// A Backend wrapper that routes neural network evaluations to one of two
// inner backends based on the double-move ability state of the position:
//   - Positions where at least one player still has the double-move ability
//     are evaluated by the "variant" backend (the network being trained).
//   - Positions where both players have exhausted their abilities are
//     evaluated by the "standard" backend (a pretrained standard lc0 network).
//
// This is transparent to the search code — it sees a single Backend*.
class DualBackend : public Backend {
 public:
  // Both backends are shared — tournament.cc uses shared_ptr for dedup.
  DualBackend(std::shared_ptr<Backend> variant_backend,
              std::shared_ptr<Backend> standard_backend);

  BackendAttributes GetAttributes() const override;
  std::unique_ptr<BackendComputation> CreateComputation() override;
  std::optional<EvalResult> GetCachedEvaluation(
      const EvalPosition& pos) override;
  UpdateConfigurationResult UpdateConfiguration(const OptionsDict& opts) override;
  bool IsSameConfiguration(const OptionsDict& opts) const override;

 private:
  std::shared_ptr<Backend> variant_backend_;
  std::shared_ptr<Backend> standard_backend_;
};

}  // namespace lczero
