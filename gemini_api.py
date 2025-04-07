import google.generativeai as genai
import PIL.Image

# Configure the API key
genai.configure(api_key="AIzaSyACloMraQUd2plyqOVpsGTc4QeMRb54-nw")  # Replace with your actual API key

# Load the image
image = PIL.Image.open('kitty.jpg')  # Ensure the image exists in the correct path

# Select the model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")  # Use "gemini-1.5-pro" for a more powerful model

# Generate content
response = model.generate_content(["What is this image?", image])

# Print response
print(response.text)
