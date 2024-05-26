from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return "Bruh"

app.run(host = "0.0.0.0", port = 80)
