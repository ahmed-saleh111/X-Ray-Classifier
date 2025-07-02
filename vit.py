
import torch 
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os 


def predict_image(model, processor, image_path, device = 'cpu' ):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prdicted_class_id = prediction.argmax().item()
        confidence = prediction.max().item()
    
    class_names = model.config.id2label[prdicted_class_id]

    return {
        'Diagnosis': class_names,
        'Confidence': f'{confidence * 100:.2f}%',
    }


if __name__ == "__main__":
    model = ViTForImageClassification.from_pretrained("models")
    processor = ViTImageProcessor.from_pretrained("models")
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_path = 'data_new/test/PNEUMONIA/person85_bacteria_424.jpeg'
    
    result = predict_image(model, processor, image_path)
    print(f'{result['Diagnosis']}\n{result['Confidence']}')
