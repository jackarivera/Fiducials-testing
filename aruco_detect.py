import cv2
import apriltag
from flask import Flask, Response

app = Flask(__name__)



# Create AprilTag detector
detector = apriltag.Detector()

def generate_frames():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera.")
        exit(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)

        for det in detections:
            for i in range(4):
                p0 = tuple(det.corners[i].astype(int))
                p1 = tuple(det.corners[(i+1) % 4].astype(int))
                cv2.line(frame, p0, p1, (0, 255, 0), 2)

            tag_center = tuple(det.center.astype(int))
            cv2.putText(frame, str(det.tag_id), tag_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """<html><body><img src="/video_feed" width="640" height="480"></body></html>"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)