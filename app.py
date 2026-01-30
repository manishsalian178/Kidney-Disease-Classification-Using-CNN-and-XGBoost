import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
import gradio as gr

# Disable Gradio analytics for cleaner interface
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# Classes (you can also load from class_names.json if you prefer)
classes = ['Cyst', 'Stone', 'Normal', 'Tumor']
img_size = 128

# --- Load CNN model ---
print("🔄 Loading CNN model...")
cnn_model = tf.keras.models.load_model("saved_models/kidney_disease_cnn_final.h5")
print("✅ CNN model loaded!")

# --- Load XGBoost model ---
print("🔄 Loading XGBoost model...")
xgb_model = joblib.load("saved_models/kidney_disease_xgb_model.pkl")
print("✅ XGBoost model loaded!")

# --- Shared preprocessing function ---
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # Convert to RGB
    img = cv2.resize(img, (img_size, img_size))    # Resize
    img = img / 255.0                              # Normalize
    img = np.expand_dims(img, axis=0)              # Add batch dimension
    return img

# --- CNN Prediction ---
def predict_cnn(image):
    img = preprocess_image(image)
    prediction = cnn_model.predict(img, verbose=0)[0]
    return {classes[i]: float(prediction[i]) for i in range(len(classes))}

# --- XGBoost Prediction (used only inside Ensemble) ---
def predict_xgb(image):
    img = preprocess_image(image)

    # Use CNN as feature extractor
    feature_extractor = tf.keras.Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.layers[-2].output  # Ensure this matches training
    )
    features = feature_extractor.predict(img, verbose=0)

    pred = xgb_model.predict_proba(features)[0]
    return {classes[i]: float(pred[i]) for i in range(len(classes))}

# --- Ensemble Prediction (Average CNN + XGBoost) ---
def predict_ensemble(image):
    cnn_pred = predict_cnn(image)
    xgb_pred = predict_xgb(image)

    final_pred = {}
    for cls in classes:
        final_pred[cls] = (cnn_pred[cls] + xgb_pred[cls]) / 2
    return final_pred

# --- Combined classify function for Gradio ---
def classify(img, choice):
    if choice == "CNN":
        return predict_cnn(img)
    elif choice == "Ensemble":
        return predict_ensemble(img)
    else:
        return {"Error": 1.0}

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Kidney Disease Classification (CT Scan)")
    gr.Markdown("Upload a CT scan image and choose between CNN or Ensemble.")

    with gr.Row():
        # Remove webcam icon by specifying only upload source
        image_input = gr.Image(type="numpy", label="Upload CT Scan Image", sources=['upload'])
        # NOTE: You have a bug here - missing "CNN" option in the radio button
        model_choice = gr.Radio(["Ensemble"], value="Ensemble", label="Choose Model")

    output = gr.Label(num_top_classes=4, label="Prediction Results")

    submit_btn = gr.Button("Predict")
    submit_btn.click(fn=classify, inputs=[image_input, model_choice], outputs=output)

# Launch with footer icons removed
demo.launch(
    show_api=False,      # Removes "Use via API" link
    show_error=False     # Removes error footer
)