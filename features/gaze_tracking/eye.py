# import math
# import numpy as np
# import cv2
# from .pupil import Pupil


# class Eye(object):
#     """
#     This class creates a new frame to isolate the eye and
#     initiates the pupil detection.
#     """

#     LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
#     RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

#     def __init__(self, original_frame, landmarks, side, calibration):
#         self.frame = None
#         self.origin = None
#         self.center = None
#         self.pupil = None
#         self.landmark_points = None

#         self._analyze(original_frame, landmarks, side, calibration)

#     @staticmethod
#     def _middle_point(p1, p2):
#         """Returns the middle point (x,y) between two points

#         Arguments:
#             p1 (dlib.point): First point
#             p2 (dlib.point): Second point
#         """
#         x = int((p1.x + p2.x) / 2)
#         y = int((p1.y + p2.y) / 2)
#         return (x, y)

#     def _isolate(self, frame, landmarks, points):
#         """Isolate an eye, to have a frame without other part of the face.

#         Arguments:
#             frame (numpy.ndarray): Frame containing the face
#             landmarks (dlib.full_object_detection): Facial landmarks for the face region
#             points (list): Points of an eye (from the 68 Multi-PIE landmarks)
#         """
#         region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
#         region = region.astype(np.int32)
#         self.landmark_points = region

#         # Applying a mask to get only the eye
#         height, width = frame.shape[:2]
#         black_frame = np.zeros((height, width), np.uint8)
#         mask = np.full((height, width), 255, np.uint8)
#         cv2.fillPoly(mask, [region], (0, 0, 0))
#         eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

#         # Cropping on the eye
#         margin = 5
#         min_x = np.min(region[:, 0]) - margin
#         max_x = np.max(region[:, 0]) + margin
#         min_y = np.min(region[:, 1]) - margin
#         max_y = np.max(region[:, 1]) + margin

#         self.frame = eye[min_y:max_y, min_x:max_x]
#         self.origin = (min_x, min_y)

#         height, width = self.frame.shape[:2]
#         self.center = (width / 2, height / 2)

#     def _blinking_ratio(self, landmarks, points):
#         """Calculates a ratio that can indicate whether an eye is closed or not.
#         It's the division of the width of the eye, by its height.

#         Arguments:
#             landmarks (dlib.full_object_detection): Facial landmarks for the face region
#             points (list): Points of an eye (from the 68 Multi-PIE landmarks)

#         Returns:
#             The computed ratio
#         """
#         left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
#         right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
#         top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
#         bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

#         eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
#         eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

#         try:
#             ratio = eye_width / eye_height
#         except ZeroDivisionError:
#             ratio = None

#         return ratio

#     def _analyze(self, original_frame, landmarks, side, calibration):
#         """Detects and isolates the eye in a new frame, sends data to the calibration
#         and initializes Pupil object.

#         Arguments:
#             original_frame (numpy.ndarray): Frame passed by the user
#             landmarks (dlib.full_object_detection): Facial landmarks for the face region
#             side: Indicates whether it's the left eye (0) or the right eye (1)
#             calibration (calibration.Calibration): Manages the binarization threshold value
#         """
#         if side == 0:
#             points = self.LEFT_EYE_POINTS
#         elif side == 1:
#             points = self.RIGHT_EYE_POINTS
#         else:
#             return

#         self.blinking = self._blinking_ratio(landmarks, points)
#         self._isolate(original_frame, landmarks, points)

#         if not calibration.is_complete():
#             calibration.evaluate(self.frame, side)

#         threshold = calibration.threshold(side)
#         self.pupil = Pupil(self.frame, threshold)


import math
import numpy as np
import cv2
from .pupil import Pupil

class Eye(object):
    """
    Lớp này tạo một frame mới để tách biệt vùng mắt và 
    bắt đầu quá trình phát hiện đồng tử (pupil detection).
    Sử dụng MediaPipe Landmarks.
    """

    # Chỉ số các điểm mắt trong MediaPipe tương ứng với 6 điểm của dlib
    # Side 0 (Mắt bên trái màn hình - Mắt phải của người)
    LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    # Side 1 (Mắt bên phải màn hình - Mắt trái của người)
    RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Trả về điểm giữa (x, y) của hai điểm"""
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    def _get_point(self, landmarks, index, width, height):
        """Chuyển đổi tọa độ chuẩn hóa của MediaPipe sang tọa độ pixel"""
        point = landmarks[index]
        return (int(point.x * width), int(point.y * height))

    def _isolate(self, frame, landmarks, points):
        """Tách riêng vùng mắt khỏi khuôn mặt."""
        height, width = frame.shape[:2]
        
        # Lấy tọa độ pixel cho các điểm landmarks
        region = np.array([self._get_point(landmarks, point, width, height) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Tạo mask để chỉ lấy vùng mắt
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cắt frame mắt (Cropping)
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        # Đảm bảo tọa độ không vượt quá giới hạn ảnh
        min_x, max_x = max(0, min_x), min(width, max_x)
        min_y, max_y = max(0, min_y), min(height, max_y)

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        h_eye, w_eye = self.frame.shape[:2]
        self.center = (w_eye / 2, h_eye / 2)

    def _blinking_ratio(self, landmarks, points, width, height):
        """Tính toán tỷ lệ nhắm mắt (Eye Aspect Ratio)."""
        p0 = self._get_point(landmarks, points[0], width, height)
        p3 = self._get_point(landmarks, points[3], width, height)
        p1 = self._get_point(landmarks, points[1], width, height)
        p2 = self._get_point(landmarks, points[2], width, height)
        p5 = self._get_point(landmarks, points[5], width, height)
        p4 = self._get_point(landmarks, points[4], width, height)

        left = p0
        right = p3
        top = self._middle_point(p1, p2)
        bottom = self._middle_point(p5, p4)

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Phát hiện, cô lập vùng mắt và khởi tạo đối tượng Pupil."""
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        height, width = original_frame.shape[:2]
        
        self.blinking = self._blinking_ratio(landmarks, points, width, height)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)