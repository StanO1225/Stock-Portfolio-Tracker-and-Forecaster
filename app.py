from flask import Flask, render_template, request
import portfolio_tracker as pt
import yfinance as yf

global pf 

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def main():
    pf = pt.Portfolio()
    script = ""
    plot1 = ""
    plot2=""    
    marketVal = 0

    if request.method == 'POST':
        data = request.form
        
        tick = data["tick"].upper()
        date = data["date"]
        shares = float(data["shares"])

        pf.add_Stock(tick, date, shares)
        
        script, div = pf.visualize_Profits()

        plot1 = div["Profit Line"]
        plot2 = div["Investment History"]
    
    return render_template("index.html", transactions=pf.getTransactions(), script=script, line_div=plot1, varea=plot2, marketVal=marketVal)

app.run(host = "0.0.0.0", port = 8000, debug=True)
