import uvicorn
import torch, os
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from contextlib import asynccontextmanager

# Define the lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer # global variables are loaded once during the application startup
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

# This defines a GET endpoint at the root URL / with a tag "authentication". 
# It redirects to the documentation page /docs using RedirectResponse.
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# Define the summarize route
@app.post("/summarize")
async def predict_route(text: str):
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

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# This block runs the application using uvicorn on host 0.0.0.0 and port 8080 if the script is executed directly.
if __name__=="__main__":
    port = int(os.environ.get("PORT", 8080))  # Render uses PORT=10000, default to 8080 for local
    uvicorn.run(app, host="0.0.0.0", port=port)