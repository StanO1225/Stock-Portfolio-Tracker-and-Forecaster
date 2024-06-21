from flask import Flask, render_template, request
import portfolio_tracker as pt
import yfinance as yf
import bokeh
import logging
global pf 

app = Flask(__name__)
pf = pt.Portfolio()


@app.route("/", methods=["GET", "POST"])
def main():
    script = ""
    plot1=""
    plot2=""
    plot3="" 
    plot4=""
    plot5=""
    error=""
    netChange = 0
    dayChange = 0

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
        plot3 = div["Portfolio Overview"]
        plot4 = div["Data Table"]
        plot5 = div["Bar"]

        netChange=pf.netChange()
        dayChange=pf.netChange("last")

        # except Exception as e:
        #     logging.error(traceback.format_exc())            
        #     error = f"ERROR: An error occurred when processing your request."

    return render_template("index.html", script=script, line_div=plot1, varea=plot2, pie= plot3, table=plot4, bar=plot5, marketVal=f"{pf.net_worth:.2f}", numStocks=pf.numStocks(), error=error, netChange=netChange, dayChange=dayChange)    

app.run(host = "0.0.0.0", port = 8000, debug=True)
