// test_ggml_forward.cpp — first slice of the C++ ggml L0 adapter: prove we can load a
// GGUF and run a *diffusion* (bidirectional) forward through llama.cpp, then cross-check
// the result against the Python lab.
//
// open-dCoder is a Qwen2ForCausalLM, so it converts to a stock (causal) Qwen2 GGUF — but
// llama_set_causal_attn(ctx, false) makes the decode bidirectional regardless, and we apply
// the Dream-family shift (logits for position p come from row p-1) ourselves. The first
// masked slot of "...return a +" must predict " b", exactly as the lab's
// test_dcoder_adapter.py::test_shifted_head_yields_meaningful_prediction asserts.
//
// usage: test_ggml_forward <model.gguf>
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_ggml_forward <model.gguf>\n");
        return 1;
    }
    const char * model_path = argv[1];
    const llama_token MASK = 151665;  // open-dCoder <M>
    const char * prompt = "def add(a, b): return a +";
    const int n_mask = 4;

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 64; cp.n_ubatch = 64;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "failed to create context\n"); llama_model_free(model); return 1; }

    llama_set_causal_attn(ctx, false);  // the diffusion forward: fully bidirectional

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Tokenize the prompt (no BOS, matching the lab's raw tok.encode for Qwen2).
    std::vector<llama_token> toks(128);
    int n = llama_tokenize(vocab, prompt, (int) std::strlen(prompt), toks.data(),
                           (int) toks.size(), /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) { std::fprintf(stderr, "tokenize failed (buffer)\n"); return 1; }
    toks.resize(n);
    const int prompt_len = n;

    // Board = prompt + masked slots.
    std::vector<llama_token> board = toks;
    for (int i = 0; i < n_mask; ++i) board.push_back(MASK);
    const int n_board = (int) board.size();

    // One batch over the whole board; request logits at every position.
    llama_batch batch = llama_batch_init(n_board, 0, 1);
    batch.n_tokens = n_board;
    for (int i = 0; i < n_board; ++i) {
        batch.token[i] = board[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
    const float * logits = llama_get_logits(ctx);

    // First masked position; Dream-family shift => read row (p-1).
    const int p = prompt_len;
    const float * row = logits + static_cast<size_t>(p - 1) * n_vocab;
    int argmax = 0;
    float best = row[0];
    for (int t = 1; t < n_vocab; ++t)
        if (row[t] > best) { best = row[t]; argmax = t; }

    char piece[256];
    int np = llama_token_to_piece(vocab, argmax, piece, sizeof(piece), 0, false);
    if (np < 0) np = 0;
    piece[np] = '\0';

    std::printf("prompt_len=%d n_board=%d  first-mask pos=%d -> token %d = '%s'\n",
                prompt_len, n_board, p, argmax, piece);
    const bool ok = (std::string(piece) == " b");
    std::printf("RESULT: %s (expected ' b', matching the lab's open-dCoder shifted-head test)\n",
                ok ? "PASS" : "MISMATCH");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 2;
}
