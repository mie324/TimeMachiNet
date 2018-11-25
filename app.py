from flask import Flask, render_template

app = Flask(__name__)


# Route for home or index
@app.route('/')
def index():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
