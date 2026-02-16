# RAG Engine SaaS (Gemini)

> Production-grade RAG system with multi-model LLM support, streaming, and web search

Production-oriented RAG project:
- Backend (FastAPI + SQLite): upload files, chunk + index, semantic retrieval (Gemini/OpenAI/Claude/Ollama embeddings), chat Q&A with citations
- Frontend (React + Mantine): session-based workspace, drag/drop upload, chat, and source cards
- **NEW**: Streaming responses, multi-model support, web search augmentation

## 🚀 Features

### Multi-Model LLM Support
- **Google Gemini** (default)
- **OpenAI GPT-4o / GPT-4o-mini**
- **Anthropic Claude 3.5 Sonnet / Haiku**
- **Ollama** (local models like Llama3.2, Mistral)

### Advanced RAG
- Hybrid search (vector + BM25)
- Cross-encoder reranking (optional)
- **NEW**: Web Search augmentation (like Perplexity)
- Streaming responses via Server-Sent Events (SSE)
- Citation tracking

### Production Features
- Multi-provider embeddings
- Session-based workspace
- File management (upload, delete, list)
- Message history
- Health checks & metrics

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLite** - Lightweight database
- **Python** 3.11+

### Frontend
- **React** + TypeScript
- **Mantine** - React UI components
- **Vite** - Build tool

### AI/ML
- **Google Gemini** - Primary LLM
- **OpenAI** - GPT models (optional)
- **Anthropic** - Claude models (optional)
- **Ollama** - Local models (optional)

## 📖 Documentation

- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Development Guide](docs/development.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🚦 Quick Start

### 1. Backend

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: GEMINI_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY)
```

### 2. Run Backend

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` and set the backend URL to `http://localhost:8000`.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider: `gemini`, `openai`, `anthropic`, `ollama` | `gemini` |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2` |
| `GEMINI_MODEL` | Gemini model | `gemini-2.5-flash` |
| `OPENAI_MODEL` | OpenAI model | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Anthropic model | `claude-3-5-haiku-20241022` |
| `ENABLE_STREAMING` | Enable SSE streaming | `true` |
| `ENABLE_WEB_SEARCH` | Enable web search | `false` |
| `TAVILY_API_KEY` | Tavily API key for web search | - |

### Example .env

```bash
# Choose one provider
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key

# Or OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_key

# Or Anthropic
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=your_anthropic_key

# Or Ollama (local)
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.2

# Optional: Enable streaming
ENABLE_STREAMING=true

# Optional: Enable web search (Perplexity-like)
# ENABLE_WEB_SEARCH=true
# TAVILY_API_KEY=your_tavily_key
```

## 📡 API Endpoints

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions` | Create new session |
| GET | `/api/sessions/{id}/files` | List session files |
| DELETE | `/api/sessions/{id}` | Delete session |

### Files

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions/{id}/files` | Upload files |
| DELETE | `/api/sessions/{id}/files/{file_id}` | Delete file |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions/{id}/chat` | Chat (non-streaming) |
| POST | `/api/sessions/{id}/chat/stream` | Chat with streaming |

### Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List available models |

## 🔄 Streaming Chat

Use Server-Sent Events for real-time responses:

```javascript
const response = await fetch(`${backendUrl}/api/sessions/${sessionId}/chat/stream`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Your question",
    temperature: 0.2,
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.delta) {
        // Stream in the response
      }
      if (data.done) {
        // Final response with citations
      }
    }
  }
}
```

## 🎯 Usage Tips

1. **Choose your LLM**: Set `LLM_PROVIDER` to gemini, openai, anthropic, or ollama
2. **Enable streaming**: Set `ENABLE_STREAMING=true` for real-time responses
3. **Web search**: Enable `ENABLE_WEB_SEARCH=true` for Perplexity-like experience
4. **Local models**: Use Ollama to run models locally without API costs

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

See CONTRIBUTING.md for details.
