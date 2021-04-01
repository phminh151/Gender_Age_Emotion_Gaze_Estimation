from flask import Flask, render_template, Response
import cv2
import sys
import numpy

app = Flask(__name__)

def get_frame():
    # video_path=0
    video_path = "C:\\Users\\Admin\\Downloads\\Video\\eye2.mp4"
    camera=cv2.VideoCapture(video_path) #this makes a web cam object

    while True:
        retval, im = camera.read()
        imgencode=cv2.imencode('.jpg',im)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

    del(camera)

@app.route('/vid')
def vid():
     return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost',port=5000, debug=True, threaded=True)