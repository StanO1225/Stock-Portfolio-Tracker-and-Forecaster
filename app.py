from flask import Flask, render_template, request
import portfolio_tracker as pt
import yfinance as yf

global pf 
global hasGraph

app = Flask(__name__)
pf = pt.Portfolio()

@app.route("/", methods=["GET", "POST"])
def main():
    hasGraph = False

    if request.method == 'POST':
        data = request.form
        print(data)
        tick = data["tick"].upper()
        date = data["date"]
        shares = float(data["shares"])

        pf.add_Stock(tick, date, shares)
        hasGraph = True

        

    return render_template("index.html", transactions= pf.getTransactions(), ifGraph = hasGraph)

app.run(host = "0.0.0.0", port = 80)
