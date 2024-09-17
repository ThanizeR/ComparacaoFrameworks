import sqlite3

# Função para visualizar os dados do banco de dados
def view_feedback_data(database_file):
    # Conectar ao banco de dados
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    
    # Executar uma consulta SQL para selecionar todos os dados da tabela feedback
    cursor.execute("SELECT * FROM feedback")
    
    # Recuperar os resultados da consulta
    rows = cursor.fetchall()
    
    # Imprimir os dados
    for row in rows:
        print(row)
    
    # Fechar a conexão com o banco de dados
    conn.close()

# Chamar a função para visualizar os dados do banco de dados
feedback_database_file = 'feedbackGradio.db'
view_feedback_data(feedback_database_file)
