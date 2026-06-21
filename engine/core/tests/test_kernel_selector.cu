// test_kernel_selector.cu — parity gate for the §4.3 kernel integration: the CUDA
// KernelCommitSelector must produce the SAME commit set as the backend-free CpuCommitSelector
// on identical logits. Token ids + committed positions must match exactly (the kernel's greedy
// path is bit-validated against reference.py); confidences agree within float32-vs-float64
// epsilon. Runs on the GPU; build with -DCLOZE_BUILD_CUDA=ON.
#include "cloze/selector.hpp"
#include "cloze/selector_kernel.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace cloze;

namespace {

// Deterministic logits: one clear peak per row, peak height rising with the row so the
// confidence ordering (and thus top-k) is unambiguous. n rows, V vocab.
ForwardResult make_logits(int n, int V) {
    ForwardResult fwd;
    fwd.n_requested = n;
    fwd.vocab = V;
    fwd.logits.assign(static_cast<size_t>(n) * V, 0.0f);
    for (int r = 0; r < n; ++r) {
        const int peak = (r * 37 + 5) % V;
        fwd.logits[static_cast<size_t>(r) * V + peak] = 3.0f + 0.4f * r;  // rising peaks
    }
    return fwd;
}

// True if the two selections commit the same (pos, token) pairs in the same order, with
// confidences within epsilon.
bool same(const Selection& a, const Selection& b) {
    if (a.commit.size() != b.commit.size()) return false;
    for (size_t i = 0; i < a.commit.size(); ++i) {
        if (a.commit[i].pos != b.commit[i].pos) return false;
        if (a.commit[i].token_id != b.commit[i].token_id) return false;
        if (std::fabs(a.commit[i].confidence - b.commit[i].confidence) > 1e-4) return false;
    }
    return true;
}

int failures = 0;

void check_case(const char* name, int n, int V, const StepContext& ctx, int topk) {
    ForwardResult fwd = make_logits(n, V);
    std::vector<int> want;
    for (int i = 0; i < n; ++i) want.push_back(10 + i);  // arbitrary board positions

    const Selection cpu = CpuCommitSelector().select(fwd, want, ctx, topk);
    const Selection gpu = KernelCommitSelector().select(fwd, want, ctx, topk);

    const bool ok = same(cpu, gpu);
    std::printf("  %-22s n=%d V=%d topk=%d -> cpu %zu / gpu %zu commits : %s\n",
                name, n, V, topk, cpu.commit.size(), gpu.commit.size(), ok ? "PASS" : "MISMATCH");
    if (!ok) {
        ++failures;
        for (size_t i = 0; i < cpu.commit.size() || i < gpu.commit.size(); ++i) {
            if (i < cpu.commit.size())
                std::printf("    cpu[%zu] pos=%d tok=%d conf=%.5f\n", i, cpu.commit[i].pos,
                            cpu.commit[i].token_id, cpu.commit[i].confidence);
            if (i < gpu.commit.size())
                std::printf("    gpu[%zu] pos=%d tok=%d conf=%.5f\n", i, gpu.commit[i].pos,
                            gpu.commit[i].token_id, gpu.commit[i].confidence);
        }
    }
}

}  // namespace

int main() {
    std::printf("test_kernel_selector: CpuCommitSelector vs KernelCommitSelector parity\n");
    // Fixed-k selection at a realistic vocab.
    check_case("fixed k=2", 6, 2000, StepContext{0, 4}, 2);
    check_case("fixed k=1", 8, 50000, StepContext{1, 4}, 1);
    // Quota mode (topk<0): k = ceil(n/steps_remaining).
    check_case("quota step0/4", 8, 2000, StepContext{0, 4}, -1);   // ceil(8/4)=2
    check_case("quota step3/4", 5, 2000, StepContext{3, 4}, -1);   // ceil(5/1)=5 (all)
    // k larger than n -> clamp to n (all rows commit).
    check_case("k>n clamps", 4, 2000, StepContext{0, 4}, 100);

    if (failures == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d case(s) FAILED\n", failures);
    return 1;
}
