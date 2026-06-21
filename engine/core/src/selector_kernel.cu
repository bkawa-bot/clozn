// selector_kernel.cu — KernelCommitSelector: drive the confidence-select CUDA kernel from the
// generate loop's selector seam. Compiled by nvcc so cudaMalloc/Memcpy + the kernel link
// directly. Greedy + MaxProb + TopK — the kernel's validated deterministic path.
#include "cloze/selector_kernel.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "confidence_select.cuh"

namespace cloze {

namespace {

void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}

// k resolution mirrors confidence_topk exactly (quota ceil(n/steps_remaining) or fixed k),
// then clamped to [1, n] since the kernel's TopK mode needs k_commit >= 1.
int resolve_k(int n, const StepContext& ctx, int topk) {
    int take;
    if (topk < 0) {
        const int rem = ctx.steps_remaining();
        if (rem < 0) throw std::invalid_argument("quota mode (topk<0) requires ctx.steps_total");
        take = (n + rem - 1) / rem;
    } else {
        take = topk;
    }
    if (take > n) take = n;
    if (take < 1) take = 1;
    return take;
}

// RAII for a device buffer so a throw mid-sequence never leaks.
struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t bytes) { cu(cudaMalloc(&p, bytes), "malloc"); }
    ~DevBuf() { if (p) cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

}  // namespace

Selection KernelCommitSelector::select(const ForwardResult& fwd, const std::vector<int>& want,
                                       const StepContext& ctx, int topk) const {
    const int n = fwd.n_requested;
    const int V = fwd.vocab;
    if (static_cast<int>(want.size()) != n)
        throw std::invalid_argument("want size != logits rows");
    if (n == 0) return Selection{};
    const int k = resolve_k(n, ctx, topk);

    DevBuf d_logits(sizeof(float) * static_cast<size_t>(n) * V);
    DevBuf d_tok(sizeof(int32_t) * n);
    DevBuf d_conf(sizeof(float) * n);
    DevBuf d_sel(sizeof(int32_t) * n);
    DevBuf d_nsel(sizeof(int32_t));

    if (fwd.device_resident) {
        // Zero-copy (DESIGN §4.3): the logits already live on the GPU. Gather the requested
        // source rows on-device (D2D) into a packed [n, vocab] buffer — no full-vocab D2H/H2D
        // crosses the bus; only the ~2*n outputs come back. Handles gappy/shifted masked sets.
        if (static_cast<int>(fwd.device_src_rows.size()) != n)
            throw std::invalid_argument("device_src_rows size != logits rows");
        for (int i = 0; i < n; ++i) {
            const int srow = fwd.device_src_rows[i];
            float* dst = static_cast<float*>(d_logits.p) + static_cast<size_t>(i) * V;
            if (srow >= 0) {
                if (srow >= fwd.device_n_rows) throw std::out_of_range("device_src_rows out of range");
                cu(cudaMemcpyAsync(dst, fwd.device_logits + static_cast<size_t>(srow) * V,
                                   sizeof(float) * V, cudaMemcpyDeviceToDevice, nullptr),
                   "D2D gather");
            } else {
                // -1: the frozen boundary row lives on the host (one row); copy it up into its slot.
                if (static_cast<int>(fwd.boundary_row.size()) != V)
                    throw std::invalid_argument("boundary_row size != vocab for device gather");
                cu(cudaMemcpyAsync(dst, fwd.boundary_row.data(), sizeof(float) * V,
                                   cudaMemcpyHostToDevice, nullptr), "H2D boundary row");
            }
        }
    } else {
        // Host path (portable / CPU build): upload the [n, vocab] host logits once.
        cu(cudaMemcpy(d_logits.p, fwd.logits.data(), sizeof(float) * static_cast<size_t>(n) * V,
                      cudaMemcpyHostToDevice), "H2D logits");
    }

    ConfidenceSelectParams p{};
    p.n_masked = n;
    p.vocab = V;
    p.temperature = 0.0f;  // greedy: argmax + softmax confidence
    p.top_p = 1.0f;        // disabled
    p.confidence = ConfidenceKind::MaxProb;
    p.mode = SelectMode::TopK;
    p.k_commit = k;
    p.tau = 0.0f;
    p.min_commit = 1;
    p.rng_seed = 0;
    ConfidenceSelectOutputs o{static_cast<int32_t*>(d_tok.p), static_cast<float*>(d_conf.p),
                              static_cast<int32_t*>(d_sel.p), static_cast<int32_t*>(d_nsel.p)};

    confidence_select(static_cast<const float*>(d_logits.p), p, o, /*stream=*/nullptr);
    cu(cudaDeviceSynchronize(), "sync");
    cu(cudaGetLastError(), "kernel");

    std::vector<int32_t> tok(n), sel(n);
    std::vector<float> conf(n);
    int32_t nsel = 0;
    cu(cudaMemcpy(tok.data(), d_tok.p, sizeof(int32_t) * n, cudaMemcpyDeviceToHost), "D2H tok");
    cu(cudaMemcpy(conf.data(), d_conf.p, sizeof(float) * n, cudaMemcpyDeviceToHost), "D2H conf");
    cu(cudaMemcpy(&nsel, d_nsel.p, sizeof(int32_t), cudaMemcpyDeviceToHost), "D2H nsel");
    if (nsel > 0)
        cu(cudaMemcpy(sel.data(), d_sel.p, sizeof(int32_t) * nsel, cudaMemcpyDeviceToHost),
           "D2H sel");

    // Map the kernel's selected ROW indices back to board positions via `want`. The kernel
    // returns them ascending by row, so want[row] is ascending too (commit order).
    Selection out;
    out.commit.reserve(nsel);
    for (int i = 0; i < nsel; ++i) {
        const int row = sel[i];
        out.commit.push_back(Candidate{want[row], static_cast<int>(tok[row]),
                                       static_cast<double>(conf[row])});
    }
    return out;
}

}  // namespace cloze
