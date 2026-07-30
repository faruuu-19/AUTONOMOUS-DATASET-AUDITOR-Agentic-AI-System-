## Replit Frontend + Python Backend

This backend now exposes the API expected by the Replit frontend:

- `POST /api/audit/start`
- `GET /api/audit/:id/status`
- `GET /api/audit/:id/report`

### 1) Start the Python backend

```bash
cd backend
python -m pip install -r requirements.txt
python api_server.py
```

Backend runs on `http://localhost:5000` by default.

### 2) Run the frontend in client mode

```bash
cd frontend
npm install
npm run dev:client
```

This uses Vite and proxies `/api` calls to `http://localhost:5000`.

### 3) Optional: Serve built frontend from Python backend

```bash
cd frontend
npm run build
```

Then open `http://localhost:5000` and Flask will serve `frontend/dist/public`.
