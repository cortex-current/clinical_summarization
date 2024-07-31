# app.py 
from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from clinical_summary.pipeline.prediction import PredictionPipeline

app = FastAPI()

# This defines a GET endpoint at the root URL / with a tag "authentication". 
# It redirects to the documentation page /docs using RedirectResponse.
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# This defines a GET endpoint at /train. When accessed, it attempts to run a training script (main.py). 
# If successful, it returns a "Training successful !!" message. 
# If an error occurs, it catches the exception and returns an error message.
@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# This defines a POST endpoint at /predict. 
# It expects some text input, processes it using a PredictionPipeline object, and returns the predicted text. 
# If an error occurs during the prediction, it raises the exception.
@app.post("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e

# This block runs the application using uvicorn on host 0.0.0.0 and port 8080 if the script is executed directly.
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)