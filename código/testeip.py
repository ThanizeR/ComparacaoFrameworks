import PyPDF2

# Substitua 'seu_arquivo.pdf' pelo caminho do seu arquivo PDF
with open('/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/código/contrato.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    
    # Extraia os metadados
    info = reader.metadata

    # Exiba os metadados
    for key, value in info.items():
        print(f'{key}: {value}')