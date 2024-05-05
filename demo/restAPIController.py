from flask import Flask, jsonify, request

app = Flask(__name__)

# Definir una ruta para manejar las consultas
@app.route('/query', methods=['POST'])
def handle_query():
    # Obtener el texto de la consulta desde la solicitud
    query_text = request.json.get('query')
    
    # Realizar la consulta utilizando el query_engine
    response = query_engine.query(query_text)
    
    # Devolver la respuesta como JSON
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(debug=True)
