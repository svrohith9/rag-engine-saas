# Contributing

Thanks for contributing.

## Local development

```bash
# backend
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# frontend
cd ../frontend
npm install
```

## Validation before PR

- Backend: `pytest`
- Frontend: `npm run build`
- Docs updated when API/config changes

## PR checklist

- [ ] Focused scope
- [ ] Clear description + rationale
- [ ] Validation output included
- [ ] No secrets committed
