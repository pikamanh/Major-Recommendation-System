from deepface import DeepFace
from features.gaze_tracking import GazeTracking
import cv2
import time
import pandas as pd

class Camera:
    def __init__(self):
        self.gaze = GazeTracking()
        self.cap = cv2.VideoCapture(0)

    def is_run(self):
        if self.cap.isOpened():
            return True
        
        return False

    def run(self, skip_frame=10):
        num_frame = 0
        dataframe = {
            'Timestamps': [],
            'Emotional': [],
            'Eye focused': []
        }
        start_time = time.time()

        while True:
            num_frame += 1
            elapsed_time = round(time.time() - start_time, 2)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if num_frame % skip_frame == 0:
                #Gaze Tracking
                self.gaze.refresh(frame)
                # frame = self.gaze.annotated_frame()
                is_focus = "Focus" if self.gaze.is_center() else "Not Focus"

                #Emotional
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']

                dataframe["Timestamps"].append(elapsed_time)
                dataframe["Emotional"].append(emotion)
                dataframe["Eye focused"].append(is_focus)

            cv2.imshow("Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                df = pd.DataFrame(dataframe)
                df.to_csv("results/result_analysis.csv", index=False)

                self.cap.release()
                cv2.destroyAllWindows()
                break