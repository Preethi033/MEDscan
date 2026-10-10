
import torch

CLASSES = ["Normal", "Pneumonia"]

def predict(model, image_tensor, device):

    model.eval()

    with torch.no_grad():

        logits = model(
            image_tensor.unsqueeze(0).to(device)
        )

        probs = torch.softmax(
            logits,
            dim=1
        )[0]

    pred_idx = probs.argmax().item()

    return {
        "label": CLASSES[pred_idx],
        "probs": {
            "Normal": float(probs[0]),
            "Pneumonia": float(probs[1])
        }
    }
