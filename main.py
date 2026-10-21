
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Artificts"
BASE_DIR = Path(__file__).resolve().parent


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Validate model files
if not (MODEL_DIR / "config.json").is_file():
    raise FileNotFoundError(
        f"config.json missing in {MODEL_DIR}"
    )

if not (MODEL_DIR / "model.safetensors").is_file():
    raise FileNotFoundError(
        f"model.safetensors missing in {MODEL_DIR}"
    )

# Load tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR)
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    str(MODEL_DIR),
    use_safetensors=True
)

model.to(device)
model.eval()

# API
app = FastAPI(
    title="English to Hinglish Translator",
    version="1.0.0"
)


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.get("/")
def home():
    return {
        "message": "English to Hinglish Translator API",
        "status": "running",
        "device": str(device)
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


@app.post("/translate")
def translate(request: TranslationRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    try:
        # IMPORTANT: Use the same prefix as training/testing
        input_text = "translate English to Hinglish: " + text

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True
            )

        translation = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return {
            "input": text,
            "translation": translation.strip()
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(exc)}"
        )

app.mount(
    "/ui",
    StaticFiles(directory=str(BASE_DIR / "static"), html=True),
    name="frontend"
)
