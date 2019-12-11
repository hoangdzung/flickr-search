from flask import Flask, render_template, request, abort
from backend import process, content2dglid
from tqdm import tqdm 

app = Flask(__name__)

terms = []
users = []
groups = []

for ntype, content in tqdm(content2dglid, desc="Read keywords"):
   if ntype == 'term':
      terms.append(content)
   elif ntype == 'user':
      users.append(content)
   elif ntype == 'group':
      groups.append(content)

groups = sorted(groups)
users = sorted(users)
terms = sorted(terms)

@app.route('/')
def student():
   return render_template('home.html', terms=terms, users=users,groups=groups)

@app.route('/result',methods = ['POST'])
def result():
   if request.method == 'POST':
      result = request.form
      term = ""
      user = ""
      group = ""
      if "term" in result: term = result["term"]
      if "user" in result: user = result["user"]
      if "group" in result: group = result["group"]

      print(result)
      data = process(result)
      return render_template("result.html", data = data, term = term, user=user, group=group,\
         terms=terms, users=users,groups=groups)
   else:
      abort(405)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8889, debug = False)
