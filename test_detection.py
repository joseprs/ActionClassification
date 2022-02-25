import io
from deepface.detectors import FaceDetector

frame_path = '00555.jpg'
face_detector = FaceDetector.build_model('retinaface')
image = io.imread(frame_path)
face_loc = FaceDetector.detect_faces(face_detector, 'retinaface', image)
print(face_loc)