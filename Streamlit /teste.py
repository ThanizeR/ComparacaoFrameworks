import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

def predict_malaria(img):
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.astype(np.float32) / 255.0
    img = img.reshape((1,36,36,3))
    model = load_model("models/malaria.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    return pred_class, pred_prob

def predict_pneumonia(img):
    img = img.convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.astype(np.float32) / 255.0
    img = img.reshape((1,36,36,1))
    model = load_model("models/pneumonia.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    return pred_class, pred_prob

def page_malaria():
    st.header("Previsão de Malária")
    uploaded_file = st.file_uploader("Faça o upload de uma imagem para previsão de malária", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Imagem enviada", use_column_width=True)
            pred_class, pred_prob = predict_malaria(img)
            if pred_class == 1:
                st.success(f"Previsão: Infectado - Probabilidade: {pred_prob*100:.2f}%")
            else:
                st.success(f"Previsão: Não está infectado - Probabilidade: {pred_prob*100:.2f}%")
        except Exception as e:
            st.error(f"Erro ao prever Malária: {str(e)}")

def page_pneumonia():
    st.header("Previsão de Pneumonia")
    uploaded_file = st.file_uploader("Faça o upload de uma imagem para previsão de pneumonia", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Imagem enviada", use_column_width=True)
            pred_class, pred_prob = predict_pneumonia(img)
            if pred_class == 1:
                st.success(f"Previsão: Pneumonia - Probabilidade: {pred_prob*100:.2f}%")
            else:
                st.success(f"Previsão: Saudável - Probabilidade: {pred_prob*100:.2f}%")
        except Exception as e:
            st.error(f"Erro ao prever Pneumonia: {str(e)}")

def main(selected_page):
    if selected_page == "Malaria":
        page_malaria()
    elif selected_page == "Pneumonia":
        page_pneumonia()

menu = st.sidebar.radio(
    "Navegação",
    ["🦟 Detecção Malária", "🫁 Detecção Pneumonia"]
)

def get_selected_page(menu):
    if menu == "🦟 Detecção Malária":
        return "Malaria"
    elif menu == "🫁 Detecção Pneumonia":
        return "Pneumonia"

selected_page = get_selected_page(menu)

if __name__ == "__main__":
    main(selected_page)
