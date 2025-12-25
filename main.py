from cores.camera import Camera
from services.score_service import Score

if __name__ == "__main__":
    camera = Camera()
    score = Score()

    camera.run()
    emotion_score = score.emotional_scores()
    print(emotion_score)