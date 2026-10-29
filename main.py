import cv2
import mediapipe as mp
import math


options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_faces=1
)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
timestamp = 0

memes ={
     "smile": cv2.resize(cv2.imread("meme/CharlieKirk.png"), (400,400)),
     "neutral": cv2.resize(cv2.imread("meme/epstein.png"), (400,400)),
     "shocked": cv2.resize(cv2.imread("meme/ronaldo.png"), (400,400))
}

def distance(a, b, width, height):
      aposition = a.x * width, a.y *height
      bposition = b.x* width, b.y * height
      return math.dist(aposition, bposition)
def mouth_open(face, width, height):
    return distance(face[13], face[14], width, height)
def smile_width(face, width,height):
    return distance(face[61], face[291], width, height)
def eye_open(face, width, height):
     return distance(face[159], face[145], width, height)
def brow_raise(face, width, height):
     return distance(face[105], face[159], width, height)
def detect_expression(mo,sw, eo, br):
    if sw >= 0.4:
         return "smile"
    elif mo >= 0.1 and eo >= 0.1:
         return "shocked"
    else:
         return "neutral"



cap = cv2.VideoCapture(0) 
while True:
    success, frame = cap.read()
    if not success:
        print("camera error")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)
    timestamp += 33
    result = landmarker.detect_for_video(mp_image, timestamp)
    if result.face_landmarks:
        check_point = [13, 14, 61, 291]
        face = result.face_landmarks[0]
        height, width,color = frame.shape

       

        face_height = distance(face[10], face[152], width, height)
        mo = round((mouth_open(face, width, height) / face_height), 1)
        eo = round((eye_open(face, width, height) / face_height), 1)
        sw = round((smile_width(face, width, height) /face_height), 1)
        br = round((brow_raise(face, width, height) / face_height),1)
    
        expression = detect_expression(mo,sw, eo, br)
        if expression in memes:
             cv2.imshow("Meme", memes[expression])

        
        

    
 

    cv2.imshow("FujifilmXM5", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()