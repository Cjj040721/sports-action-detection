import os.path
from flask import Flask, jsonify
from flask_cors import CORS
from TestPth import getResult
from counting import get_action_count_result


app = Flask(__name__)
# 启用跨域支持
CORS(app, resources={
    r"/detect": {"origins": "*"},
    r"/upload_csv": {"origins": "*"},  # 新增上传文件的跨域支持
    r"/export_csv": {"origins": "*"}
})


@app.route('/detect', methods=['GET'])
def detect():
    mat_path = r"..\MATLAB code\adcSampleAll.mat"
    # 返回简短的 JSON 响应
    while True:
        if os.path.exists(mat_path):
            try:
                result = getResult()
                result = get_action_count_result(mat_path, result)
                os.remove(mat_path)
                return jsonify(result)
            except Exception as e:
                print(f"Error processing file: {e}")
                return jsonify({"error": "Error processing file"}), 500
        else:
            return jsonify({"prediction": "No"})


if __name__ == '__main__':
    app.run(debug=True)
