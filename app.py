from flask import Flask, render_template, request, redirect
import os
import use_model

input_dir = './static/test/input/pic'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = input_dir


# Route for home or index
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        return redirect('http://localhost:5000/test')


@app.route('/test', methods=['GET'])
def test():
    if request.method == 'GET':
        use_model.evaluate()
    return render_template('test.html')


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
