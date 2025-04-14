from flask import Flask, request, jsonify
import google.generativeai as genai
import PIL.Image
import io
import threading
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configure the API key (replace "my-key" with your actual API key)
genai.configure(api_key="hehe")

# Select the model (use "gemini-1.5-flash" or "gemini-1.5-pro" as needed)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def show_image_temporarily(image):
    # Display the image using matplotlib for 3 seconds
    plt.imshow(image)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(3)  # Display for 3 seconds
    plt.close()

@app.route('/detect', methods=['POST'])
def generate_content():
    # Ensure an image file is part of the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Please attach an image with the key "image".'}), 400

    file = request.files['image']
    try:
        # Open the image from the incoming file stream
        image = PIL.Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': 'Invalid image file', 'message': str(e)}), 400

    # Show the image on the server side in a non-blocking manner
    # Use image.copy() to avoid threading issues with PIL image objects
    threading.Thread(target=show_image_temporarily, args=(image.copy(),)).start()
    
    # Generate content based on the image using the provided code
    response = model.generate_content([
        "What is this image? describe in very short and if the object is dangerous for a visually impaired person, add a 'DANGER', at the end, so the format of your answer should be: 'description, danger', if not dangerous say 'SAFE'", 
        image
    ])
    
    # Log the generated response to the server console
    print("Generated response:", response.text)
    
    # Return only the generated text to the client as JSON
    return jsonify({'text': response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
