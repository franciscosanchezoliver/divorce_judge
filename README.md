## Divorce Judge (LangGraph)

Simple (non-binding) judge to analyze divorce judgments in PDF using language models (OpenAI or Ollama) and an orchestrated flow with LangGraph.

The idea is to explore how AI can support sensitive domains like law by producing structured summaries and evaluations of judgments, always with a strong disclaimer: this is not legal advice and does not replace professional human judgment.

### Requirements

- Python 3.10+
- `virtualenv` (optional but recommended)

Main dependencies (listed in `requirements.txt`):

- `langchain`, `langgraph`
- `langchain-openai`, `langchain-ollama`
- `pydantic`, `python-dotenv`
- `pypdf`

### Installation

```bash
git clone https://github.com/franciscosanchezoliver/divorce_judge.git
cd divorce_judge

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment variables

To use OpenAI models:

```bash
export OPENAI_API_KEY="your_api_key"
```

To use Ollama models:

- Have `ollama` installed and running locally.
- Download the model you want to use (for example `qwen2.5:3b`).

You can use a `.env` file in the project root (it's in `.gitignore`) and set your keys there:

```env
OPENAI_API_KEY=your_api_key
```

### Usage

The entry point is `src/main.py`. Example:

```bash
python -m src.main \
    --file /path/to/your/judgment.pdf \
    --provider ollama \
    --model qwen2.5:3b
```

Or with OpenAI:

```bash
python -m src.main \
    --file /path/to/your/judgment.pdf \
    --provider openai \
    --model gpt-4o-mini
```

The program:

- Converts the PDF to text.
- Extracts relevant facts (parties, custody, alimony, etc.).
- Evaluates the coherence and completeness of the judgment.
- Generates a structured JSON with:
  - `summary`
  - `extracted_facts`
  - `evaluation`

Final output is logged using `logging` (level `INFO`).

### Limitations and legal notice

- The model can make mistakes, hallucinate data, or miss relevant information.
- This system does not replace a professional legal analysis nor constitute legal advice.
- The project is intended as an educational/exploratory exercise on using AI in legal contexts.

