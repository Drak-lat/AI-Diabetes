from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_voting import predict_soft

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Đặt key bất kỳ, miễn là không để trống

COLUMN_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    input_data = {}
    error = None
    if request.method == 'POST':
        input_data = {name: request.form.get(name, '') for name in COLUMN_NAMES}
        try:
            values = [float(input_data[name]) for name in COLUMN_NAMES]
            X = np.array([values])
            pred = predict_soft(X)
            session['result'] = {
                'input_data': input_data,
                'detail': {
                    'logistic': round(pred['logistic']*100, 2),
                    'knn': round(pred['knn']*100, 2),
                    'rf': round(pred['rf']*100, 2),
                    'soft_voting': round(pred['soft_voting']*100, 2)
                }
            }
            return redirect(url_for('result'))
        except Exception as e:
            error = f"Lỗi dữ liệu nhập vào: {e}"
    return render_template('index.html', input_data=input_data, error=error)

@app.route('/result')
def result():
    result = session.get('result')
    if not result or 'detail' not in result:
        # Nếu truy cập thẳng /result mà không có dữ liệu sẽ trả về trang nhập
        return redirect(url_for('index'))
    return render_template('result.html', input_data=result['input_data'], detail=result['detail'])

if __name__ == '__main__':
    app.run(debug=True)
