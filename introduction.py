from flask import Flask, jsonify, render_template
import json

app = Flask(__name__)

# 预加载 GeoJSON 文件
geojson_files = {
    '1': 'buildings_geojson.geojson',
    '2': 'land_geojson.geojson',
    '3': 'soil_maxvorstadt_geojson.geojson',
    '4': 'points_geojson.geojson',
    '5': 'lines_geojson.geojson',
}
geojson_data = {}

for key, filepath in geojson_files.items():
    with open('static/geojson' + '/' + filepath, 'r',encoding='utf-8') as file:
        geojson_data[key] = json.load(file)

@app.route('/')
def index():
    return render_template('introduction.html')

@app.route('/geojson/<key>')
def send_geojson(key):
    return jsonify(geojson_data.get(key, {}))

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
