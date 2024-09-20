import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse
from transformers import AutoTokenizer

app = FastAPI()

# Initialize tokenizer and model
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        # Load the model and tokenizer
        model = torch.load('model_cpu_friendly.pth', map_location=torch.device('cpu'))
        tokenizer = AutoTokenizer.from_pretrained('Falconsai/medical_summarization')
        model.eval()  # Set the model to evaluation mode
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# This defines a GET endpoint at the root URL / with a tag "authentication". 
# It redirects to the documentation page /docs using RedirectResponse.
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict")
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
                                     num_beams=8)  # Optionally use beam search for better quality)
        
        # Decode the generated output
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# This block runs the application using uvicorn on host 0.0.0.0 and port 8080 if the script is executed directly.
if __name__=="__main__":
    port = int(os.environ.get("PORT", 8080))  # Render uses PORT=10000, default to 8080 for local
    uvicorn.run(app, host="0.0.0.0", port=port)