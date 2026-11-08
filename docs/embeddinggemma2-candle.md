# EmbeddingGemma 2 — Native Candle text encoder (experimental)

Branch: `embeddinggemma2`, **based directly on `main`**.
Independent from `mirror_orch` and all ASD workflow / GPU-integration branches.

## Current scope

* Rust/Candle-only text path for `google/embeddinggemma-2`.
* Parses `EmbeddingGemma2Model` `config.json`; model tensors under `language_model.*`.
* Bidirectional full/sliding attention with per-layer Q/K/V shapes; additive PLE; Gemma 4-style RMSNorm; per-token projection; mean pooling; Matryoshka 128/256/512/768 L2 output.
* **F32**, including on Pascal `sm_61`; F16 explicitly rejected.
* No Python, PyTorch, virtualenv, remote embedding API, or LLM server.
* No image/audio tower yet. This code has not been parity-certified against the Google checkpoint.

## Usage

```bash
git fetch origin
git switch -c embeddinggemma2 --track origin/embeddinggemma2

cargo test -p candle-transformers embedding_gemma2::tests --lib
cargo run --release -p candle-examples --example embedding-gemma2 -- \
  --text 'Question: quel port TCP utilise Upsilon ?' --dims 256
```

This retrieves Hugging Face files through the Rust `hf-hub` dependency. If the model requires terms acceptance or authorization, configure the Hugging Face token through the normal hub mechanism.

Run with an existing local checkpoint directory, with `config.json`, `tokenizer.json` and safetensors shards:

```bash
cargo run --release -p candle-examples --example embedding-gemma2 -- \
  --local-dir /path/to/embeddinggemma-2 \
  --text 'Passage: sm_61 fallback cuDNN.' --dims 256 --json
```

Only after baseline CPU parity is established, try CUDA F32 (device 0):

```bash
cargo run --release -p candle-examples --features cuda --example embedding-gemma2 -- \
  --cuda --local-dir /path/to/embeddinggemma-2 \
  --text 'Passage: sm_61 fallback cuDNN.' --dims 256
```

### Important current limitations

1. **Do not assume parity yet.** Compare identical token IDs, preprocessing/instructions, per-token states, pooling and normalized vectors to the pinned Google Transformers model.
2. Inputs longer than 8192 tokens, empty inputs, batches other than one, or F16 are disallowed.
3. Sentence Transformers query/document prompt templates are **not** automatically prepended; pass the exact desired text (including task prompt) to `--text`.
4. Sliding attention currently allocates a dense (sequence x sequence) F32 mask; use shorter sequences until optimized.
5. Google full multimodal `EmbeddingGemma2Model` has image/audio modules, but this standalone Candle path loads only `language_model.*`.
6. Do not rewrite or reuse the old MiniLM 384D HMD checkpoints/indexes: EmbeddingGemma 2 vectors are a separate embedding space.
7. Real safetensors load/inference and GPU speed **must be verified**, not inferred from passing syntactic CI.

## Validation sequence

1. `cargo check -p candle-transformers --lib`
2. `cargo check -p candle-examples --example embedding-gemma2`
3. `cargo test -p candle-transformers embedding_gemma2::tests --lib`
4. Load pinned real weights on CPU F32.
5. Compare exact tokenizer sequence and cosine similarity to a trusted reference at 768, 512, 256, 128 dimensions, across French, code, varied lengths and prompts.
6. Confirm index metadata (model, revision, dimensionality, prompt policy, dtype, normalization).

No changes are required in `mirror_orch` or `mistral.rs` for this stage.
