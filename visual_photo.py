import streamlit as st
from PIL import Image
from io import BytesIO


from langchain_google_vertexai import VertexAI
import vertexai
vertexai.init(project="saraswati-ai", location="us-central1")
  
# llm = VertexAI(model_name="gemini-pro")


from vertexai import generative_models
from vertexai.generative_models import GenerativeModel
model = GenerativeModel(model_name="gemini-1.0-pro-vision")


# img  = "https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/a21e8a6fcd4bd0d5e9e883c9b3c579383139cbc4ef5df81ce331b3494a23f8ed?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27swin_transformer_architecture.png%3B+filename%3D%22swin_transformer_architecture.png%22%3B&response-content-type=image%2Fpng&Expires=1713884643&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMzg4NDY0M319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy9hMjFlOGE2ZmNkNGJkMGQ1ZTllODgzYzliM2M1NzkzODMxMzljYmM0ZWY1ZGY4MWNlMzMxYjM0OTRhMjNmOGVkP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=Y30wvQNwNaSQe1I8w2VemkkL8UZBmbYv36G7ndjAKl8SGnT8SBHBWNp8cTc0e6OG9v0Z7Few3yO-VR4H4Wa1BiLRZBz2VsuK-vs0mMtD7QnyOjCMRHb27qYYIEy5uSLv3AK1y-GYw0iaao4oazSyUoVXBnweycIWK0CMgngFtY5xttTtb7zIbD6q8q9XLcfT5VjGkNjcf4ZRmAzguWEBiU%7EKTtXR9pXc1BfpU312Fi%7EoH3uQPkhc3Vj6QJ3wh-44QDlcwklOZdqjyLeab1f9MM2xN3gDVEvnkoGb3SXqkHE8aAL8ro1nymX4sPY6aRUXEkHAwOcPG0A1igMKyATXEg__&Key-Pair-Id=KVTP0A1DKRTAX"

import base64

# Define a function to process the image and generate explanation
def process_image(image):
    # Here, you would implement your logic to process the image and generate explanation
    # For demonstration purpose, let's just return a placeholder explanation
    image = image.resize((500,500))
    img_byte_arr = BytesIO()
    # image.save(img_byte_arr, format=image.format)

    img_byte_arr = img_byte_arr.getvalue()

    # Encode the byte string as base64
    encoded_img = base64.b64encode(img_byte_arr).decode()
    response = model.generate_content(["What is this?", encoded_img])

    return response.explanation

# Main Streamlit app
def main():
    st.title("Image Explanation Agent")

    # File uploader widget to let the user upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button to process the image and generate explanation
        if st.button("Explain"):
            # Process the image and generate explanation
            explanation = process_image(image)
            # Display the explanation
            st.write("Explanation:")
            st.write(explanation)

if __name__ == "__main__":
    main()
