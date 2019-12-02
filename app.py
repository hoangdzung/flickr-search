from flask import Flask, render_template, request, abort
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('home.html')

@app.route('/result',methods = ['POST'])
def result():
   if request.method == 'POST':
      result = request.form
      print(result)
      images = process(result)
      return render_template("result.html", images = enumerate(images))
   else:
      abort(405)

def process(result):
   # for 
   return ['static/images/556.jpg','static/images/Hanoi_Adobe.jpg', 'static/images/IMG_1266.jpg', 'static/images/COOL-WALLPAPER-7037-21909636.jpg']

if __name__ == '__main__':
   app.run(debug = True)