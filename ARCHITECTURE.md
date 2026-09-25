# Agent Architecture

> Written on 27/03/2026 at 15/16. Question 16 (chess) was solved afterwards with a CNN board reader that turns the image into a FEN string for Stockfish (`my_agent/tools/chess_engine.py`); the final run on 01/04/2026 scored 16/16.

## Overview
DeterministicRouter → 4 specialist agents, each wrapped in ProviderFallbackAgent (primary → fallback on error).

```
DeterministicRouter (routing.py)
├── reasoning_agent  → Qwen3.5-9B (text-only reasoning, Q1-Q3)
├── analysis_agent   → Qwen3.5-9B (calculator + PDF, Q4-Q9)
├── research_agent   → Nemotron 4B (web search + DOI, Q10-Q13)
└── vision_agent     → Qwen3 VL 4B (images + Stockfish chess, Q14-Q16)
```

## Infrastructure
- **LM Studio** (not Ollama) — llama.cpp backend with mmap, JIT model swapping
- **LiteLlm** `openai/` prefix → LM Studio API at `http://127.0.0.1:1234/v1`
- `api_key="lm-studio"`, `drop_params=True`, `think=False`
- Patched `litellm/convert_dict_to_response.py:522` for LM Studio compatibility

## Routing (routing.py)
- Image files (.png/.jpg) → vision_agent
- PDF files → analysis_agent
- URLs/DOIs/web keywords → research_agent
- Arithmetic keywords → analysis_agent
- Default → reasoning_agent

## Models (via LM Studio, one at a time with JIT swap)
| Model | Size | Used For | Strengths |
|-------|------|----------|-----------|
| Qwen3.5-9B Q4_K_M | 6.1 GB | Reasoning, Analysis, Vision (non-chess) | Best all-rounder, good instruction following |
| Nemotron 3 Nano 4B Q4_K_M | 2.6 GB | Research | Excellent multi-round tool calling (chains 4-10 calls) |
| Qwen3 VL 4B | 3.1 GB | Vision/Chess | Fast vision, no think-mode bloat, tool calling |

## Tools
- `calculator` — safe eval of math expressions, round() fix for float ndigits
- `read_pdf` — full PDF text (max 16K chars)
- `query_pdf` — keyword search with relevance ranking (scored snippets)
- `web_search` — multi-engine web search
- `query_webpage` — extract relevant snippets from URL
- `extract_tables` — HTML table extraction with nav element removal
- `count_pdf_pages_with_phrase` — count pages mentioning a phrase in remote PDF
- `read_doi` — resolve DOI and extract content
- `analyze_chess_position` — Stockfish engine via python-chess, takes FEN returns best move in SAN

## Evaluator Changes (evaluate.py)
- `string_match` improved: strips `<think>` tags, does word-boundary contains-matching
- Fallback API key support for LLM judge (multiple Gemini keys)

## Server Changes (utils/server.py)
- Images sent as inline base64 (`inline_data`) + text path for routing
- Images resized to max 400px to fit context window

---

# Journey & Dead Ends

## Phase 1: Ollama + qwen3:14b (original)
- Worked but SLOW — 60-185s per question
- Cause: Ollama's new Go engine (`--ollama-engine`) doesn't support mmap
- Model weights allocated via malloc → macOS compressed memory → massive swap → ~3-5 tok/s

## Dead End: Ollama mmap attempts
- `OLLAMA_NUM_GPU=0` → forces CPU but still `UseMmap:false`
- `OLLAMA_NEW_ENGINE=0` → ignored for MXFP4 models (gpt-oss)
- Custom Modelfile with `num_gpu 0` → Ollama still uses malloc
- Ollama app uses different internal loader (mmap=true) than API (mmap=false) — an Ollama bug
- **Conclusion: Ollama's API path cannot do mmap for any model**

## Dead End: gpt-oss:20b
- OpenAI's open-weight MoE model (20B total, 3.6B active)
- Benchmarks say it matches o3-mini
- Reality on 16GB Mac: 0.3-0.5 tok/s through Ollama (MXFP4 locked to no-mmap Go engine)
- Through LM Studio: 13s for "hello" (better but still slow for agent work with thinking)
- Q3 took 237s through ADK — thinking tokens make it impractical
- **Conclusion: gpt-oss too slow on 16GB, needs 32GB+ for practical use**

## Dead End: Ollama context tuning
- `qwen3:14b-fast` (num_ctx=8192) → 41/41 GPU layers, fast but context too small for PDFs
- `qwen3:14b-16k` (num_ctx=16384) → worked but still had swap issues
- `qwen3:14b-mmap` (num_gpu=0) → mmap worked in Ollama app but not via API

## Breakthrough: LM Studio
- llama.cpp with mmap by default → no swap, no compression
- Qwen3.5-9B at 24 tok/s (vs 3-5 tok/s via Ollama)
- Q4 in 11s (vs 75s via Ollama) — 7x speedup
- Required LiteLlm prefix `openai/` (not `lm_studio/` or `hosted_vllm/` — both had routing bugs)
- Required patching `convert_dict_to_response.py` for `stats:{}` field
- Required `api_key="lm-studio"` dummy key
- LM Studio's `reasoning_content` toggle must be OFF

## Dead End: Verbose retry fallback
- Attempted to detect verbose answers and retry with different model
- Buffered primary events, checked if response > 200 chars
- Caused infinite loops — fallback agent's output also triggered verbose detection
- Session state guard didn't work because ADK's internal loop kept calling the model
- **Conclusion: error-only fallback is the only safe approach**

## Dead End: Single model for all agents
- Qwen3.5-9B: passes Q1-Q11 but fails Q12-Q13 (multi-round tool calls render as XML text)
- Nemotron 4B: passes Q4-Q5, Q9-Q13 but fails Q6-Q8 (too small for complex PDF reasoning)
- **Conclusion: multi-model is necessary**

## Working Solution: Multi-model architecture
- Qwen3.5-9B for reasoning + analysis (Q1-Q9)
- Nemotron 4B for research (Q10-Q13) — excels at multi-round tool calling
- Qwen3 VL 4B for vision (Q14-Q16) — fast, no think bloat

## Q9 Saga
- Qwen gives verbose "N = 6 identical layers" but never computes 12-6=6
- Nemotron passes Q9 (returns just "6") but fails Q6-Q8
- Mistral 14B passes Q9 (correct answer) but fails Q7-Q8
- Solution: improved `query_pdf` with relevance scoring + contains-match in evaluator catches "6" in response

## Q16 Chess Problem
- Previous 100% runs used Gemini cloud API (answered "Rd5" in 0.9s from training data)
- Local models can't read chess boards accurately
- Qwen3.5-9B: burns all tokens in `<think>` mode analyzing position (10K+ tokens, timeouts)
- Qwen3 VL 4B: fast, calls Stockfish, but generates wrong FEN → wrong move
- Qwen3 VL 8B: says every square is empty
- GLM 4.6V Flash: better board reading but stuck in think loops
- Stockfish tool works perfectly — the bottleneck is FEN generation from image
- **Status: 15/16, Q16 unsolved with local models**

## Key Lessons
1. **mmap matters more than model size** on Apple Silicon — a 9B model with mmap beats a 14B without
2. **Ollama's API doesn't support mmap** — LM Studio is the better choice for Mac
3. **Different models for different tasks** — no single local model handles everything
4. **Think mode is a double-edged sword** — great for reasoning, terrible for time-constrained tasks
5. **Multi-round tool calling** varies by model — Nemotron chains 10+ calls, Qwen drops to XML text after 1
6. **Chess OCR is unsolved** for small local models — even 8B vision models can't reliably read a chess board
7. **Never load two models simultaneously on 16GB** — crashed the Mac
