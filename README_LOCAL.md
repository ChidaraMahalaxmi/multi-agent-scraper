# Run `multiagentscraper` In VS Code (Local)

## 1) Open terminal in this folder
Use terminal path:
`.../aiScraper -dataparadigm/multiagentscraper`

## 2) Create virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

## 3) Install dependencies
```powershell
pip install -r requirements_local.txt
```

If you want Playwright fallback scraping:
```powershell
playwright install chromium
```

## 4) Ensure Ollama is running locally
```powershell
ollama serve
```

Pull model if needed:
```powershell
ollama pull llama3.2:3b
```

## 5) Run pipeline from VS Code terminal
```powershell
python run_local.py --goal "Extract PROTACs and linkers from 2025 in table format"
```
By default this creates:
- `result.json` (full pipeline state + output)
- `result_table.csv` (tabular extracted rows)

Search source is DuckDuckGo (web pages), not arXiv/bioRxiv APIs.

## Useful options
```powershell
python run_local.py `
  --goal "Extract PROTAC findings in table format" `
  --max-iterations 1 `
  --max-papers 8 `
  --max-chunks 20 `
  --ollama-chat-timeout 35 `
  --model "llama3.2:3b" `
  --model-planner "llama3.2:1b" `
  --model-filter "llama3.2:1b" `
  --model-retrieval "llama3.2:1b" `
  --model-extractor "llama3.2:3b" `
  --ollama-host "http://127.0.0.1:11434" `
  --save-json "result.json" `
  --save-table "result_table.csv"
```

Recommended defaults:
- planner/filter/retrieval: `llama3.2:1b` (faster)
- extraction: `llama3.2:3b` (better quality)

Enable Playwright only when needed:
```powershell
python run_local.py --goal "..." --use-playwright
```

Enable deeper semantic retrieval scoring (slower):
```powershell
python run_local.py --goal "..." --retrieval-llm-scoring --max-llm-chunks-per-doc 2
```

## Troubleshooting
- If slow: reduce `--max-papers` and `--max-chunks`.
- If no extraction: check `errors` in output JSON summary.
- If Ollama connection fails: verify `ollama serve` and host URL.
