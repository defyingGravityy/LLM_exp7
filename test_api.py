"""Quick automated check of the testing checklist (uses whatever BACKEND is in .env)."""
from fastapi.testclient import TestClient
from app import app

c = TestClient(app)
print("health      ", c.get("/health").status_code, c.get("/health").json())
for label, q in [("supported", "How do I request a refund?"),
                 ("unsupported", "Do you sell home insurance?")]:
    r = c.post("/ask", json={"question": q})
    j = r.json()
    print(f"{label:12}", r.status_code, "| grounded:", j.get("grounded"),
          "| latency_ms:", j.get("latency_ms"), "\n             ", j.get("answer", j)[:120])
print("empty       ", c.post("/ask", json={"question": "   "}).status_code)
print("too long    ", c.post("/ask", json={"question": "a" * 1001}).status_code)
