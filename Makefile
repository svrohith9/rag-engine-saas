.PHONY: backend-dev frontend-dev

backend-dev:
	cd backend && \
		python3.11 -m venv .venv && \
		source .venv/bin/activate && \
		pip install -r requirements.txt && \
		python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend-dev:
	cd frontend && npm install && npm run dev
