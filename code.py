from flask import Flask
from flask import request
from MotionDetector import motiondect

app = Flask(__name__)

@app.route('/model', methods = ['GET', 'POST', 'DELETE'])
def mlmodel():
    if request.method == 'POST':
        print("Posted file: {}".format(request.files['file']))
        file = request.files['file']
        data = motiondect(file)
        print(data)

if __name__ == '__main__':
    app.run(port=5000)