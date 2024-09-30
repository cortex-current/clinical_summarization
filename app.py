import uvicorn
import torch, os
from fastapi import FastAPI, HTTPException, Request, Form
from starlette.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from contextlib import asynccontextmanager
# Jinja2 Templates: Used for rendering HTML, displaying the input text, and showing the summarization result.

# Define the lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    try:
        # Load the Hugging Face model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained('manoramak/finetuned-clinical-summarizer')
        tokenizer = AutoTokenizer.from_pretrained('manoramak/finetuned-clinical-summarizer')
        model.eval()  # Set the model to evaluation mode
        print("Model and tokenizer loaded successfully")
        yield
    except Exception as e:
        raise RuntimeError(f"Error loading model or tokenizer: {e}")
    finally:
        print("Shutting down application")

# Create the FastAPI app with the lifespan event handler
app = FastAPI(lifespan=lifespan)

# Serve the static directory for any static files (e.g., CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory using Jinja2
# Used for rendering HTML, displaying the input text, and showing the summarization result.
templates = Jinja2Templates(directory="templates")

# Route to serve the index HTML page
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the summarize route for the form POST submission
@app.post("/summarize", response_class=HTMLResponse)
async def predict_route(request: Request, text: str = Form(...)):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded")

    try:
        # Preprocess the input text
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=False)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate the output
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     max_length=256,  # Set max_length to desired value
                                     min_length=50,   # Optionally set min_length if you want a minimum length
                                     length_penalty=0.8,  # Optionally adjust length_penalty to control the length
                                     num_beams=8)  # Optionally use beam search for better quality

        # Decode the generated output
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Render the result back to the HTML page
        return templates.TemplateResponse("index.html", {
            "request": request, "prediction": prediction, "input_text": text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
