from PIL import Image
import requests
import torch
import socket
import gradio as gr

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def predict(inp_img: Image, text):
    """Predicts the probabilities of a list of comma-separated text for a given image.

    Args:
        image_url: The URL of the image.
        text: A comma-separated list of text.

    Returns:
        A list of probabilities for each text.
    """
    image = inp_img
    input_text = text.split(",")
    inputs = processor(text=input_text, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = torch.nn.functional.softmax(logits_per_image[0], dim=0) # we can take the softmax to get the label probabilities
    confidences = {input_text[i]: float(probs[i]) for i in range(len(input_text))}

    return confidences

# app = gradio.Interface(
#     fn=predict,
#     inputs=[
#         gradio.inputs.Image(label="Image"),
#         gradio.inputs.Textbox(label="Text", value="cat,dog,bird"),
#     ],
#     outputs=[
#         gradio.outputs.Textbox(label="Probabilities"),
#     ],
#     title="CLIP Text Prediction",
#     description="This app predicts the probabilities of a list of comma-separated text for a given image using CLIP.",
# )

if __name__ == "__main__":
    gr.Interface(
        fn=predict, inputs=[gr.Image(type="pil"), gr.Textbox(label="Text") ], outputs=gr.Label(num_top_classes=10),
		title=f"CLIP Text Prediction from {socket.gethostname()}",
  description="This app predicts the probabilities of a list of comma-separated text for a given image using CLIP."
    ).launch(server_name="0.0.0.0")
