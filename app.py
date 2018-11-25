from flask import Flask, render_template
from neural_network import main
app = Flask(__name__)


# Route for home or index
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/test')
def test():
    return render_template('test.html')


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)

