# LangChain SQLite Examples

Examples demonstrating LangChain integration with SQLite for SQL query generation and LLM responses.

## Setup

```bash
cd $(mktemp -d)
git clone https://github.com/RyanGreenup/langchain_sql_example
just init
just run
```

## Usage

Run basic SQL query generation:
```bash
just run
```

Run agent-based queries:
```bash
just run-agent
```

Check for type errors:
```bash
just check
```

## Files

- `file.ts` - Basic SQL query generation using `generate_sql_query` function
- `agent.ts` - Agent-based SQL querying with formatted output
- `examples_of_langchain_db_llm` - Advanced graph-based query processing examples

## Requirements

- Node.js
- pnpm
- ANTHROPIC_API_KEY environment variable
