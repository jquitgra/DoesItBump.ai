from flask import Flask
from flask import request
from MotionDetector import motiondect

app = Flask(__name__)

@app.route('/model', methods = ['GET', 'POST', 'DELETE'])
def mlmodel():
    if request.method == 'POST':
        fileasmp4 = ""
        file = request.files['file'] 
        f = file.read() 
        r = motiondect(f)
        print(r)


if __name__ == '__main__':
    app.run(port=5000)