from flask import Flask, request, jsonify
import mysql.connector
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading

app = Flask(__name__)

# 🔹 Conectar ao banco de dados
def conectar_bd():
    try:
        return mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="eduardo20694",
            database="inteligencia_artificial"
        )
    except mysql.connector.Error as err:
        print(f"Erro ao conectar ao banco de dados: {err}")
        return None

# 🔹 Carregar modelo de embeddings
modelo_embedding = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# 🔹 Carregar perguntas e respostas do banco
def carregar_dados():
    conn = conectar_bd()
    if conn is None:
        return [], []
    
    cursor = conn.cursor()
    cursor.execute("SELECT pergunta, resposta FROM conhecimento WHERE ativo = TRUE")
    dados = cursor.fetchall()
    conn.close()
    
    if not dados:
        return [], []
    
    perguntas, respostas = zip(*dados)
    return list(perguntas), list(respostas)

# 🔹 Atualizar embeddings dinamicamente
def atualizar_embeddings():
    global perguntas, respostas, perguntas_embeddings
    perguntas, respostas = carregar_dados()
    if perguntas:
        perguntas_embeddings = modelo_embedding.encode(perguntas)
    else:
        perguntas_embeddings = np.array([])

# 🔹 Iniciar embeddings ao rodar o servidor
perguntas, respostas = [], []
perguntas_embeddings = np.array([])
atualizar_embeddings()

# 🔹 Rota para testar conexão com o banco de dados
@app.route('/testar_conexao', methods=['GET'])
def testar_conexao():
    conn = conectar_bd()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE();")
        db = cursor.fetchone()
        conn.close()
        return jsonify({"message": f"Conexão bem-sucedida ao banco de dados: {db[0]}"})
    return jsonify({"erro": "Erro ao conectar ao banco de dados."}), 500

# 🔹 Encontrar resposta usando similaridade de cosseno
def encontrar_resposta(pergunta):
    if perguntas_embeddings.size == 0:
        return "Ainda não há perguntas cadastradas."
    
    embedding_pergunta = modelo_embedding.encode([pergunta])
    similaridades = cosine_similarity(embedding_pergunta, perguntas_embeddings)[0]
    
    indice_mais_similar = np.argmax(similaridades)
    maior_similaridade = similaridades[indice_mais_similar]
    
    if maior_similaridade < 0.6:
        return "Não entendi sua pergunta. Poderia reformular?"
    
    return respostas[indice_mais_similar]

# 🔹 Rota para testar API
@app.route('/teste', methods=['GET'])
def teste():
    return jsonify({"message": "API está funcionando!"})

# 🔹 Rota principal
@app.route('/', methods=['GET'])
def raiz():
    return jsonify({"message": "Bem-vindo à API de Inteligência Artificial!"})

# 🔹 Rota para responder perguntas
@app.route('/pergunta', methods=['POST'])
def pergunta():
    data = request.get_json()
    pergunta_usuario = data.get('pergunta', '').strip()
    
    if not pergunta_usuario:
        return jsonify({"erro": "Por favor, envie uma pergunta válida."}), 400
    
    resposta = encontrar_resposta(pergunta_usuario)
    return jsonify({"resposta": resposta})

# 🔹 Rota para recarregar embeddings manualmente
@app.route('/atualizar_dados', methods=['POST'])
def atualizar_dados():
    thread = threading.Thread(target=atualizar_embeddings)
    thread.start()
    return jsonify({"message": "Atualização de dados iniciada."})

if __name__ == '__main__':
    app.run(debug=True)
