# Local multi-model agent (AISO x ML6 agentic lab)

An agent that answers a 16-question benchmark (reasoning, maths, PDFs, web research and images) using only local models on a 16 GB MacBook Pro (M2 Pro). Final score: **16/16**.

Built by [Artem Mamedov](https://www.linkedin.com/in/artem-mamedov) in March-April 2026, starting from ML6's "Build Your Own Agent" workshop for AISO. The original workshop instructions are in [WORKSHOP.md](WORKSHOP.md).

## What I built on top of the workshop

The workshop starts from a single agent on Google's ADK with Gemini and expects about 81% after the web search milestone. I moved the agent to local models and extended it:

- **Deterministic router** ([`my_agent/routing.py`](my_agent/routing.py)): sends each question to one of four specialist agents based on its inputs (image, PDF, URL or DOI, arithmetic, or plain reasoning).
- **One model per job**: each specialist runs the local model that handled its task best in testing.
- **Fallback** ([`my_agent/fallback_agent.py`](my_agent/fallback_agent.py)): each specialist is wrapped so that an error on the primary model switches to a backup model.
- **Tools** ([`my_agent/tools/`](my_agent/tools/)): calculator, PDF reading and ranked PDF search, web search, webpage and table extraction, page counting in remote PDFs, DOI reader, image description, and a chess tool (CNN board reader to FEN, then Stockfish).
- **Evaluator fix**: strips `<think>` blocks before matching answers.

## Architecture

```
DeterministicRouter
├── reasoning_agent  → Qwen3.5-9B           plain reasoning        (Q1-3)
├── analysis_agent   → Qwen3.5-9B           calculator + PDFs      (Q4-9)
├── research_agent   → Nemotron 3 Nano 4B   web search + DOI       (Q10-13)
└── vision_agent     → Qwen3 VL 4B          images + chess         (Q14-16)
```

Models run in LM Studio (llama.cpp, one model loaded at a time) and are called through LiteLLM.

## Results

All 47 benchmark runs from 13/03 to 01/04/2026 are in [`results/`](results/). The early runs use the workshop's Gemini setup; the later ones follow the move to local models, with the score dropping and climbing back as the setup changed. The final run (01/04/2026) scored 16/16 on local models, averaging 57 seconds per question.

## What I learned

The full story, including the dead ends, is in [ARCHITECTURE.md](ARCHITECTURE.md). The short version:

1. On Apple Silicon, memory mapping matters more than model size: a 9B model loaded with mmap in LM Studio ran about 7x faster than a 14B model through Ollama's API, which could not use mmap.
2. No single small local model handled everything: Qwen3.5-9B was the best all-rounder, Nemotron 4B was far better at long chains of tool calls, and a vision model was needed for images.
3. Fallback only on errors: retrying on "bad-looking" answers caused infinite loops.

## Run it

1. Install [LM Studio](https://lmstudio.ai), download the three models above and start its server on port 1234.
2. `uv sync`
3. Copy `my_agent/.env.example` to `my_agent/.env`. Add a Google API key only if you want the LLM judge in `evaluate.py`.
4. Run the benchmark: `uv run python evaluate.py` (or `--question 16` for a single question).

The chess tool also needs `board_to_fen`, `keras` and a Stockfish binary, which are not in `pyproject.toml`.

## Credits

Workshop, benchmark and scaffolding by ML6 for AISO (see [WORKSHOP.md](WORKSHOP.md) and the git history). Everything under "What I built" is my own work.
