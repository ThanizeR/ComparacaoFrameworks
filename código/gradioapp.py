import gradio as gr
import sqlite3

# Função para criar o banco de dados e a tabela
def create_database():
    # Conectar ao banco de dados (ou criá-lo se não existir)
    conn = sqlite3.connect('feedbackGradio.db')
    cursor = conn.cursor()
    
    # Criar a tabela de feedback se ela não existir
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            navegacao INTEGER,
            design_interface INTEGER,
            elementos INTEGER,
            desempenho INTEGER,
            atraso INTEGER,
            utilidade INTEGER,
            expectativas INTEGER,
            sugestoes TEXT,
            recursos TEXT,
            comentarios TEXT,
            nome TEXT,
            email TEXT,
            telefone TEXT,
            mensagem TEXT
        )
    ''')
    
    # Fechar a conexão com o banco de dados
    conn.close()

# Função para processar e armazenar o feedback
def process_feedback(navegacao, design_interface, elementos, desempenho, atraso, utilidade, expectativas,
                     sugestoes, recursos, comentarios, nome, email, telefone, mensagem):
    # Conectar ao banco de dados
    conn = sqlite3.connect('feedbackGradio.db')
    cursor = conn.cursor()
    
    # Inserir os dados do feedback na tabela
    cursor.execute('''
        INSERT INTO feedback (navegacao, design_interface, elementos, desempenho, atraso, utilidade,
                              expectativas, sugestoes, recursos, comentarios, nome, email, telefone, mensagem)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (navegacao, design_interface, elementos, desempenho, atraso, utilidade, expectativas, sugestoes,
          recursos, comentarios, nome, email, telefone, mensagem))
    
    # Commit para salvar as alterações e fechar a conexão com o banco de dados
    conn.commit()
    conn.close()
    
    # Retornar uma mensagem de confirmação
    return "Feedback enviado com sucesso!"

# Criar o banco de dados e a tabela
create_database()


# Interface Gradio
feedback_interface = gr.Interface(
    fn=process_feedback,
    inputs=[
        gr.Slider(1, 10, value=5, step=1, label="Navegação", info="Avalie a facilidade de navegação"),
        gr.Slider(1, 10, value=5, step=1, label="Design da Interface", info="Avalie a atratividade visual da interface"),
        gr.Slider(1, 10, value=5, step=1, label="Elementos de Interface", info="Avalie a intuitividade dos elementos de interface"),
        gr.Slider(1, 10, value=5, step=1, label="Desempenho", info="Avalie o desempenho da aplicação durante o uso"),
        gr.Slider(1, 10, value=5, step=1, label="Atraso/Lentidão", info="Avalie a ocorrência de atrasos ou lentidão na aplicação"),
        gr.Slider(1, 10, value=5, step=1, label="Utilidade", info="Avalie se encontrou as informações ou funcionalidades desejadas"),
        gr.Slider(1, 10, value=5, step=1, label="Expectativas", info="Avalie se a aplicação atendeu às suas expectativas"),
        gr.Textbox(label="Sugestões de Melhoria", placeholder="Deixe suas sugestões de melhoria aqui"),
        gr.Textbox(label="Recursos Adicionais", placeholder="Indique os recursos adicionais que gostaria de ver implementados"),
        gr.Textbox(label="Comentários Adicionais", placeholder="Deixe aqui quaisquer outros comentários"),
        gr.Textbox(label="Nome", placeholder="Seu nome"),
        gr.Textbox(label="E-mail", placeholder="Seu e-mail"),
        gr.Textbox(label="Telefone", placeholder="Seu telefone"),
        gr.Textbox(label="Mensagem", placeholder="Mensagem adicional"),
    ],
    outputs="text",
    title="Feedback dos Usuários",
    description="Deixe o seu feedback sobre a aplicação:",
)

# Lançar a interface
feedback_interface.launch()
