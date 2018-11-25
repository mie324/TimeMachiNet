from flask import Flask, render_template, request
from werkzeug import secure_filename

app = Flask(__name__)


# Route for home or index
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'File uploaded successfully!'


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
