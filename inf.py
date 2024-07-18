import os
from inference import get_model
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# Constants
API_KEY = "O8csWqO4aBsCcIfx0dDf"
MODEL_ID = "skincare-detection/5"  # Replace with your specific model ID
IMAGE_PATH = "test.jpg"  # Replace with the path to your image

# Ensure API key is set
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable is not set. Please set it in your environment or .env file.")

# Load the model
model = get_model(MODEL_ID)

# Perform inference
result = model.infer(image=IMAGE_PATH)

# Print the result
print(result)

# Optionally, save the result or process it further
result.save("result.jpg")  # Replace with the desired save path
