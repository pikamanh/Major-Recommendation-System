import pandas as pd
import os

class Score:
    def __init__(self, result_analysis_filename='result_analysis.csv'):
        # Lấy path của file kết quả sau khi xem video/slide
        result_analysis_path = os.path.join(os.getcwd(), 'results', result_analysis_filename)
        self.dataframe = pd.read_csv(result_analysis_path)

        # Trọng số của từng cảm xúc
        self.emotion_weights = {
            'happy': 2.0,
            'surprise': 1.5,
            'neutral': 1.0,
            'fear': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'disgust': -0.5
        }

    def emotional_scores(self):
        # Chuyển đổi lại thành dạng số
        self.dataframe['Eye focused'] = self.dataframe['Eye focused'].map({
            'Focus': 1,
            'No Focus': 0
        })

        # Mapping và tính toán điểm hứng thú
        self.dataframe['Emotional'] = self.dataframe['Emotional'].map(self.emotion_weights)
        self.dataframe['Interest Score'] = self.dataframe['Emotional'] * self.dataframe['Eye focused']
        
        avg_interest = self.dataframe['Interest Score'].mean()
        return avg_interest