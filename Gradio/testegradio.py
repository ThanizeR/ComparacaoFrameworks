import gradio as gr
import numpy as np
from PIL import Image
from keras.models import load_model
import pickle

# Função de previsão de pneumonia
def predict_pneumonia(img):
    img = Image.fromarray(np.uint8(img))
    img = img.convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,1))
    img = img / 255.0
    model = load_model("/Users/thanizerodrigues/Desktop/ComparacaoFrameworks/models/pneumonia.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    if pred_class == 1:
        pred_label = "Pneumonia"
    else:
        pred_label = "Saudável"
    return pred_label, pred_prob

# Função de previsão de malária
def predict_malaria(img):
    img = Image.fromarray(np.uint8(img))
    img = img.resize((36,36))
    img = np.asarray(img)
    img = img.reshape((1,36,36,3))
    img = img.astype(np.float64)
    model = load_model("/Users/thanizerodrigues/Desktop/ComparacaoFrameworks/models/malaria.h5")
    pred_probs = model.predict(img)[0]
    pred_class = np.argmax(pred_probs)
    pred_prob = pred_probs[pred_class]
    if pred_class == 1:
        pred_label = "Infectado"
    else:
        pred_label = "Não está infectado"
    return pred_label, pred_prob

with open('/Users/thanizerodrigues/Desktop/ComparacaoFrameworks/models/diabetes_model.sav', 'rb') as file:
    diabetes_model = pickle.load(file)

# Função de previsão de diabetes
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    user_input = [float(x) for x in user_input]
    diab_prediction = diabetes_model.predict([user_input])
    if diab_prediction[0] == 1:
        diab_diagnosis = 'A pessoa é diabética'
    else:
        diab_diagnosis = 'A pessoa não é diabética'
    return diab_diagnosis

# Função para exibir a página de Datasets Disponíveis
def display_datasets():
    datasets = {
        "Dataset de Malária": "https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria",
        "Dataset de Pneumonia": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        "Dataset de Doenças Cardíacas": "https://github.com/siddhardhan23/multiple-disease-prediction-streamlit-app/blob/main/dataset/heart.csv",
        "Dataset de Doenças Renais": "https://www.kaggle.com/datasets/mansoordaku/ckdisease",
        "Dataset de Diabetes": "https://github.com/siddhardhan23/multiple-disease-prediction-streamlit-app/blob/main/dataset/diabetes.csv",
        "Dataset de Doenças Hepáticas": "https://www.kaggle.com/datasets/uciml/indian-liver-patient-records",
        "Dataset de Câncer de Mama": "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data"
    }

    markdown_content = "# Datasets Disponíveis\n\n"
    markdown_content += "Esta página contém links para download e visualização de datasets utilizados na aplicação.\n\n"
    
    for dataset_name, dataset_url in datasets.items():
        markdown_content += f"**{dataset_name}:** [Download {dataset_name}]({dataset_url})\n\n"
    
    return markdown_content

# Criar a interface Gradio com guias (Tabs)
tabbed_interface = gr.TabbedInterface(
    [
        # Guia 1: Página Inicial
        gr.Interface(
            fn=lambda: """
                # Bem-vindo à Aplicação de Previsão de Anomalias Médicas

                Este é um projeto de previsão de diversas anomalias médicas usando modelos de deep learning e machine learning.

                É importante observar que os modelos utilizados nesta aplicação foram obtidos de repositórios públicos na internet e, portanto, sua confiabilidade pode variar.

                Embora tenham sido treinados em grandes conjuntos de dados médicos, é fundamental lembrar que todas as previsões devem ser verificadas por profissionais de saúde qualificados.

                ## Perguntas Frequentes

                ### Como a previsão de anomalias é feita?
                A detecção de pneumonia e malária é feita usando uma rede neural convolucional (CNN), enquanto a seção de diabetes é detectada por um modelo Random Forest.

                ### Os modelos são precisos?
                Os modelos foram treinados em grandes conjuntos de dados médicos, mas lembre-se de que todas as previsões devem ser verificadas por profissionais de saúde qualificados.

                ### Qual é o propósito desta aplicação?
                Esta aplicação foi desenvolvida para auxiliar na detecção de diversas anomalias médicas em imagens de diferentes partes do corpo.

                ### Quais tipos de anomalias médicas podem ser detectadas?
                Os modelos podem detectar várias anomalias, incluindo pneumonia, malária e diabetes.
            """,
            inputs=[],
            outputs="markdown",
            title="Página Inicial",
            description="Página inicial da aplicação de previsão de anomalias médicas."
        ),
        
        # Guia 2: Previsão de Pneumonia
        gr.Interface(
            predict_pneumonia,
            inputs=gr.Image(label="Imagem para Predição de Pneumonia"),
            outputs=["text", "text"],
            title="Previsão de Pneumonia",
            description="Faça o upload de uma imagem para prever se há pneumonia."
        ),
        
        # Guia 3: Previsão de Malária
        gr.Interface(
            predict_malaria,
            inputs=gr.Image(label="Imagem para Predição de Malária"),
            outputs=["text", "text"],
            title="Previsão de Malária",
            description="Faça o upload de uma imagem para prever se há malária."
        ),

        # Guia 4: Previsão de Diabetes
        gr.Interface(
            predict_diabetes,
            inputs=[
                gr.Textbox(label="Número de Gestações"),
                gr.Textbox(label="Nível de Glicose"),
                gr.Textbox(label="Valor da Pressão Arterial"),
                gr.Textbox(label="Valor da Espessura da Pele"),
                gr.Textbox(label="Nível de Insulina"),
                gr.Textbox(label="Valor do IMC"),
                gr.Textbox(label="Valor da Função de Pedigree de Diabetes"),
                gr.Textbox(label="Idade da Pessoa")
            ],
            outputs="text",
            title="Previsão de Diabetes",
            description="Insira os dados do paciente para prever se ele tem diabetes."
        ),
        
        # Guia 5: Datasets Disponíveis
        gr.Interface(
            fn=display_datasets,
            inputs=[],
            outputs="markdown",
            title="Datasets Disponíveis",
            description="Esta página contém links para download e visualização de datasets utilizados na aplicação."
        )
    ],
    tab_names=["Página Inicial", "Previsão de Pneumonia", "Previsão de Malária", "Previsão de Diabetes", "Datasets Disponíveis"]
)

# Lançar a interface com guias
tabbed_interface.launch()
