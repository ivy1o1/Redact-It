from fastapi import FastAPI

app = FastAPI(
    title="RedactIt API",
    description="Local-first, instruction-driven document redaction.",
    version="0.1.0",
)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "redactit-api",
    }      

