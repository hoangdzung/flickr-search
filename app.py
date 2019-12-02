from flask import Flask, render_template, request, abort
from backend import process
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('home.html')

@app.route('/result',methods = ['POST'])
def result():
   if request.method == 'POST':
      result = request.form
      print(result)
      data = process(result)
      return render_template("result.html", data = data)
   else:
      abort(405)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8877, debug = False)
