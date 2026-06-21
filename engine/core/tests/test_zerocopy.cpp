// test_zerocopy.cpp — gate for the §4.3 zero-copy path. For each mode the SAME denoise is run
// THREE ways and must produce identical commits, all matching the lab golden (invariant 3):
//   (1) CpuCommitSelector            — host logits, host sample+select (the golden path)
//   (2) KernelCommitSelector + host  — host logits uploaded H2D, kernel sample+select on GPU
//   (3) KernelCommitSelector + device— logits stay on the GPU (llama_get_logits_tensor), kernel
//                                       reads them via on-device gather, no full-vocab D2H
// Two modes are gated: whole-sequence (dcoder_add.json) and semi-AR blocks with exact KV reuse
// (dcoder_add_blocks.json) — the block mode exercises the frozen boundary row on the device path
// (one host row H2D'd into the gather; the active rows stay zero-copy). Path 3 needs a CUDA llama
// with GPU offload; device_forwards() must be > 0, proving the path engaged.
//
// usage: test_zerocopy <model.gguf>
#include "cloze/generate.hpp"
#include "cloze/model_ggml.hpp"
#include "cloze/selector.hpp"
#include "cloze/selector_kernel.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace cloze;

namespace {

struct Golden {
    std::vector<int> region;  // board[p:] after the run
    std::vector<int> kept;    // generated (truncated at eos)
    std::string text;
    std::string reason;
};

// Run the three paths for one (cfg, cache) and check parity + golden + that the zero-copy path
// actually engaged. Returns true on full pass.
bool three_way(GgmlAdapter& a, const std::vector<int>& prompt_ids, int p,
               const GenerateConfig& cfg, const CacheConfig& cache, const Golden& g,
               const CpuCommitSelector& cpu, const KernelCommitSelector& ker, const char* label) {
    a.set_device_passthrough(false);
    a.reset_logits_d2h_floats();
    GenerateResult r1 = generate(a, prompt_ids, cfg, cache, &cpu);
    const long long host_d2h = a.logits_d2h_floats();

    a.set_device_passthrough(false);
    GenerateResult r2 = generate(a, prompt_ids, cfg, cache, &ker);

    a.reset_device_forwards();
    a.reset_logits_d2h_floats();
    a.set_device_passthrough(true);
    GenerateResult r3 = generate(a, prompt_ids, cfg, cache, &ker);
    const long long devf = a.device_forwards();
    const long long dev_d2h = a.logits_d2h_floats();

    auto region = [&](const GenerateResult& r) {
        return std::vector<int>(r.board.begin() + p, r.board.end());
    };
    auto ok_golden = [&](const GenerateResult& r) {
        return region(r) == g.region && r.generated == g.kept && r.text == g.text &&
               r.reason == g.reason;
    };
    const bool g1 = ok_golden(r1), g2 = ok_golden(r2), g3 = ok_golden(r3);
    const bool agree = (r1.board == r2.board && r2.board == r3.board &&
                        r1.generated == r3.generated && r1.text == r3.text);
    const bool used = devf > 0;
    const bool ok = g1 && g2 && g3 && agree && used;

    std::printf("[%s] cpu:%s  kernel-host:%s  kernel-device:%s  agree:%s  device_forwards=%lld  %s\n",
                label, g1 ? "ok" : "X", g2 ? "ok" : "X", g3 ? "ok" : "X", agree ? "ok" : "X",
                devf, used ? "" : "(zero-copy NOT exercised!)");
    std::printf("       logits D2H over run: host %.1f MB vs device %.1f MB", host_d2h * 4.0 / 1e6,
                dev_d2h * 4.0 / 1e6);
    if (dev_d2h > 0) std::printf("  -> %.1fx fewer floats over the bus", (double)host_d2h / dev_d2h);
    std::printf("\n%s\n", ok ? "       PASS" : "       MISMATCH");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_zerocopy <model.gguf>\n");
        return 1;
    }
    const int MASK = 151665, EOS = 151643;
    const char* prompt = "def add(a, b):\n    return a +";

    GgmlAdapter adapter(argv[1], MASK, EOS, /*n_ctx=*/128, /*n_gpu_layers=*/999,
                        /*device_logits_passthrough=*/false);
    const std::vector<int> prompt_ids = adapter.encode(prompt);
    const int p = static_cast<int>(prompt_ids.size());
    const CpuCommitSelector cpu;
    const KernelCommitSelector ker;

    bool ok = true;

    // Whole-sequence (dcoder_add.json): no boundary row on the device path.
    {
        GenerateConfig cfg; cfg.max_new = 8; cfg.steps = 4; cfg.block_len = 0; cfg.topk = -1;
        Golden g;
        g.region = {293, 271, 1350, 25906, 7, 16, 11, 151643};
        g.kept = {293, 271, 1350, 25906, 7, 16, 11};
        g.text = " b\n\nprint(add(1,";
        g.reason = "eos";
        ok &= three_way(adapter, prompt_ids, p, cfg, CacheConfig{}, g, cpu, ker, "whole-seq");
    }

    // Semi-AR blocks + exact KV reuse (dcoder_add_blocks.json): exercises the frozen boundary row
    // on the device path (one host row H2D'd into the gather; the rest stay zero-copy).
    {
        GenerateConfig cfg; cfg.max_new = 8; cfg.steps = 4; cfg.block_len = 4; cfg.topk = -1;
        CacheConfig cache; cache.mode = "delta"; cache.full_refresh_every = 1;
        Golden g;
        g.region = {293, 271, 750, 526, 526, 526, 526, 526};
        g.kept = g.region;  // reason length, no eos
        g.text = " b\n\ndef int int int int int";
        g.reason = "length";
        ok &= three_way(adapter, prompt_ids, p, cfg, cache, g, cpu, ker, "block+reuse");
    }

    std::printf("\nRESULT: %s\n", ok ? "PASS" : "MISMATCH");
    return ok ? 0 : 2;
}
