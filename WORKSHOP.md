<p align="center">
  <img src="assets/logo.svg" alt="Project Logo" height="144">
</p>

# AISO Workshop - Learn How to Build an Agent by ML6

Welcome! In this 3-hour hands-on workshop you will build an AI agent from scratch using Google's Agent Development Kit (ADK). By the end, your agent will be able to reason, use tools, read documents, and search the web.

<p align="center">
  <img src="assets/agent.webp" alt="Agent Illustration" height="144">
</p>

## The Big Picture

*In the era of ChatGPT, everyone expects AI to answer any question — instantly, flawlessly, and magically. But modern AI models still lack the **tools and capabilities** to tackle the most complex problems: digging through PDFs, searching the web for up-to-date facts, performing precise calculations, and combining it all through multi-step reasoning.*

*Today you will fix that. You will build an **AI agent** — an LLM in a loop that can call tools, interpret their output, and decide what to do next — and watch it get progressively smarter as you give it new abilities.*

**What is an Agent?**

An [agent](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) is an LLM with the ability to invoke tools (Python functions — which in turn can invoke APIs) and use the output of these tools to generate better, more informed answers.

## How the Workshop Works

We will work through **5 milestones**. After each one you can run the evaluation benchmark and watch your agent's accuracy climb. At the end of each one, we will push a solution to this repository so you can compare it with yours and continue building.

| | Milestone | What you'll do | Estimated time |
|---|---|---|---|
| 0 | **Setup** | Clone, install, get the UI running | ~15 min |
| 1 | **Your first agent** | Configure a bare-bones agent and chat with it | ~15 min |
| 2 | **Calculator tool** | Give your agent the ability to do math | ~20 min |
| 3 | **PDF reader tool** | Let your agent extract information from PDFs | ~30 min |
| 4 | **Web search tool** | Connect your agent to the internet | ~40 min |
| 5 | **Free time — go wild** | Multi-agent setups, image tools, better prompts… | remaining time |

### Tracking your progress

A benchmark of 16 questions is included in `benchmark/questions.json`. Some require reasoning, some require files, some require the web. As you add tools, more questions become answerable.

```bash
# Run the full benchmark
uv run python evaluate.py

# Run a single question (1-indexed)
uv run python evaluate.py --question 1
```

You can find an overview of which questions require a specific tool here:

| Tool | Question indices |
|---|---|
| None (reasoning) | 1-3 |
| Calculator | 4-6 |
| PDF reader | 7-9 |
| Web search | 10-13 |
| Image tools | 14-16 |

| After milestone | Expected accuracy |
|---|---|
| 1 -- Base agent | ~19% |
| 2 -- Calculator | ~38% |
| 3 -- PDF reader | ~56% |
| 4 -- Web search | ~81% |

### Catching up

After each milestone we will push a solution branch to this repository so you can compare your approach with ours. If you fall behind, you can check out a solution branch and continue from there.

```bash
# Fetch the latest solution branches
git fetch origin

# Compare your code with a solution
git diff main..origin/solution/milestone-2 -- my_agent/

# Or check out a solution to continue from there
git checkout -b my-work origin/solution/milestone-3
```

---

## Milestone 0 — Setup

### Prerequisites

- Python 3.10 or higher
- A Google API key (we will provide one)

### 1. Fork and clone the repository

Go to <https://github.com/ml6team/AISO-workshop> and click **Fork** (top right) to create
your own copy. This keeps your work private and separate from other participants.

```bash
git clone https://github.com/<your-username>/AISO-workshop
cd AISO-workshop
```

### 2. Install uv (Python package manager)

**macOS/Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or via Homebrew (macOS):

```bash
brew install uv
```

### 3. Configure your API key

```bash
cd my_agent
cp .local_env .env   # Windows: copy .local_env .env
```

Open `my_agent/.env` and paste your API key:

```
GOOGLE_API_KEY="your_actual_api_key_here"
```

### 4. Install dependencies

```bash
cd ..   # back to project root
uv sync
```

### 5. Launch the UI

```bash
uv run adk web
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. You should see the ADK chat interface. In the top-left dropdown, select **my_agent** as the app, then send a test message to confirm everything works.

**Setup complete!** You are ready to start building.

---

## Milestone 1 — Your First Agent

**Goal:** Understand how the agent is configured and have a working baseline you can chat with.

Open `my_agent/agent.py`. This is the heart of your agent:

```python
root_agent = llm_agent.Agent(
    model='gemini-2.5-flash-lite',
    name='agent',
    description="A helpful assistant.",
    instruction="You are a helpful assistant that answers questions directly and concisely.",
    tools=[],
    sub_agents=[],
)
```

Take a moment to understand each parameter:

- **`model`** -- The LLM powering your agent. `gemini-2.5-flash-lite` is fast and cheap; `gemini-2.5-flash` or `gemini-2.5-pro` is more capable.
- **`instruction`** -- The system prompt. This shapes *how* your agent behaves. Be specific!
- **`tools`** -- The Python functions your agent is allowed to call. Right now it has none -- you will add them in the next milestones.

### Things to try

1. Chat with your agent in the UI — ask it a general knowledge question.
2. Try improving the `instruction` to make it more thorough (e.g., ask it to reason step-by-step).
3. Run the benchmark and note your starting accuracy:

```bash
uv run python evaluate.py
```

---

## Milestone 2 — Calculator Tool

**Goal:** Create your first custom tool and register it with the agent.

Some benchmark questions involve arithmetic. LLMs are notoriously unreliable at math — let's give your agent a proper calculator.

### Hints

1. Open `my_agent/tools/calculator.py` -- there is a stub waiting for you.
2. Implement the function: it takes an operation and two numbers, and returns the result. Think about what operations to support and how to handle edge cases (division by zero?).
3. The function's **docstring is critical** — the agent reads it to decide *when* and *how* to call the tool. Write it clearly, including argument descriptions.
4. Export your function from `my_agent/tools/__init__.py`.
5. Import it and add it to the `tools` list in `my_agent/agent.py`.
6. Update the `instruction` in `agent.py` to tell the agent *when* to use the calculator (e.g. "Use the calculator tool for all arithmetic."). A good docstring helps, but an explicit instruction is more reliable.
7. Test it in the UI: ask your agent "What is 1457 * 38?" and check if it calls your calculator tool.

### Run the benchmark

```bash
uv run python evaluate.py
```

---

## Milestone 3 — PDF Reader Tool

**Goal:** Let your agent extract information from PDF files.

Look at the benchmark questions — some reference PDF attachments (e.g., a library catalog, accommodation listings). Your agent needs a tool to read these files.

### Hints

1. Create a new file `my_agent/tools/read_pdf.py`.
2. Write a Python function that takes a file path, reads the PDF, and returns its text content.
3. Consider libraries like `PyMuPDF` (`fitz`), `pdfplumber`, or `pypdf`. You'll need to add your chosen library to the project dependencies:

   ```bash
   uv add <library-name>
   ```

4. Think about what your function should return. Raw text? A summary? How will the agent use it?
5. Remember: the **docstring** tells the agent when to use the tool. Make it clear that this tool is for PDF files.
6. Register the tool in `agent.py` just like you did with the calculator.
7. Update the `instruction` to tell the agent when to reach for this tool (e.g. "When a question references a PDF file, use the PDF reader tool with the file path provided.").
8. Test with a question that uses a PDF attachment:

   ```bash
   uv run python evaluate.py --question 7
   ```

### Run the benchmark

```bash
uv run python evaluate.py
```

---

## Milestone 4 — Web Search Tool

**Goal:** Give your agent the ability to search the web and read pages.

Several benchmark questions require real web lookups (e.g., historical facts, movie trivia, scientific publications). You will need two tools: one to **search** and one to **fetch and read** a page.

### Hints

1. Create a new file `my_agent/tools/web_search.py`. You need an actual search API. Some options:
   - **DuckDuckGo** via the `ddgs` library -- free, no API key needed (`uv add ddgs`)
   - Google Custom Search API
   - SerpAPI / Tavily
   - Or use Gemini's built-in grounding with Google Search ([ADK docs on built-in tools](https://google.github.io/adk-docs/tools/built-in-tools/))

   > **Note:** ADK's built-in `google_search` tool prevents the agent from using any other custom tools on the same agent. If you need web search *and* your calculator/PDF tools, use the `ddgs` approach instead.

2. Your search tool should return a list of results with titles, URLs, and snippets.
3. Consider creating a second tool (`fetch_webpage.py`) that takes a URL and returns the page text. The agent can then search first, then read a specific result for details.
4. Think about what to return -- the agent needs enough context to answer the question, but not so much that it gets overwhelmed. Consider truncating long pages.
5. Register both tools in `agent.py` and update the `instruction` to explain when to use each — e.g. "Use web_search to find relevant URLs, then fetch_webpage to read the content of a specific page."
6. Test with a question that requires web knowledge:

   ```bash
   uv run python evaluate.py --question 10
   ```

### Run the benchmark

```bash
uv run python evaluate.py
```

---

## Milestone 5 — Free Time: Push Your Agent Further

You've built a solid agent with tools for math, PDFs, and web search. Now it's time to get creative and see how far you can take it.

**Strategy:** Start by pushing for **100% accuracy** — check which questions your agent still gets wrong and target those first. Once you're satisfied with accuracy, look at response times and try to make your agent faster.

### Ideas to explore

- **Multi-agent architectures** — Use `sub_agents` in ADK to create specialized agents (e.g., a "researcher" and a "calculator") orchestrated by a coordinator. Check the [ADK docs on multi-agent systems](https://google.github.io/adk-docs/).
- **Image understanding** — Can you build a tool that reads and interprets images? Think about supporting multiple formats and extracting targeted information rather than generic descriptions.
- **Chess engine** — For the chess question, consider building a tool that calls a chess engine programmatically. You could install Stockfish via `uv add python-stockfish` and wrap it in a function that finds the best move for a given board position for example.
- **Better prompting** — Revisit your agent's `instruction`. Add reasoning strategies, output formatting rules, or few-shot examples.
- **Smarter tool design** — Can your tools return structured data? Can you combine tools in clever ways?
- **Try a more powerful model** — Switch to `gemini-2.5-flash` or `gemini-2.5-pro` and see if accuracy improves.
- **Analyze your failures** — Look at which benchmark questions your agent gets wrong and why. Fix the weakest link.

### Run the benchmark one last time

```bash
uv run python evaluate.py
```

How high can you get?

---

## Your Workspace

**Where you'll work: `my_agent/` folder**

- `my_agent/agent.py` — Define your agent's configuration and capabilities
- `my_agent/tools/` — Add custom tools/functions for your agent to use

**Other folders (scaffolding — do not modify):**

- `utils/` — Infrastructure code for running and evaluating agents
- `benchmark/` — Benchmark dataset and attachments
- `evaluate.py` — Evaluation script

### Project Structure

```
AISO-workshop/
├── my_agent/              # YOUR WORKSPACE
│   ├── agent.py           # Define your agent here
│   ├── tools/             # Add custom tools here
│   │   ├── __init__.py
│   │   └── calculator.py  # Stub (implement in Milestone 2)
│   ├── .local_env         # Example environment file
│   └── .env               # Your API key (create from .local_env)
├── benchmark/             # Benchmark dataset (read-only)
│   ├── questions.json     # 16 evaluation questions
│   └── attachments/       # Files referenced by some questions
├── evaluate.py            # Evaluation script
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

## Development Tips

- **Test interactively** — Use `uv run adk web` to chat with your agent and see tool calls in real time.
- **Test specific questions** — Use `uv run python evaluate.py --question <index>` to debug individual failures.
- **Test tools in isolation** — Before registering a new tool with your agent, call it directly in a Python script (`uv run python -c "from my_agent.tools.calculator import calculator; print(calculator('add', 2, 3))"`) to verify it returns what you expect.
- **Read the docs** — The [ADK documentation](https://google.github.io/adk-docs/) covers everything from tool creation to multi-agent setups.
- **Check the examples** — Browse the [ADK samples repository](https://github.com/google/adk-samples) for working examples.
- **Iterate fast** — Change something, test it, see what happens. Repeat.

## Viewing Evaluations in the Web UI

You can see evaluation runs in the chat interface:

1. Start the web UI in one terminal: `uv run adk web`
2. Run evaluations in a separate terminal: `uv run python evaluate.py`

All evaluation sessions will appear in the web UI's history.

## Troubleshooting

**"Module not found" errors:**

```bash
uv sync
```

**API key issues:**

- Make sure you copied `.local_env` to `.env` in the `my_agent/` folder
- Verify the API key is set correctly

**Port already in use:**

```bash
lsof -ti:8000 | xargs kill -9
```

## Resources

### Official Documentation

- **ADK Documentation**: [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)
- **ADK Samples**: [https://github.com/google/adk-samples](https://github.com/google/adk-samples)
- **Gemini API Docs**: [https://ai.google.dev/docs](https://ai.google.dev/docs)

### Agent Design & Best Practices

- **Building Effective Agents**: [https://www.anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents)
- **Writing Tools for Agents**: [https://www.anthropic.com/engineering/writing-tools-for-agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

### Getting Help

- **Documentation**: Almost everything you need is in the official ADK docs above
- **Stuck?** Raise your hand — ML6 engineers are here to help

Happy building!

### About ML6

ML6 is a frontier, international AI engineering company, constantly pushing the boundaries of what's possible with AI. We partner with bold leaders to turn cutting-edge AI into lasting business impact. With over a decade of proven expertise, we deliver AI that reshapes business models. AI that is reliable and secure, ensuring a lasting impact. From strategy to delivery, we don't just follow the hype—we build the future.
