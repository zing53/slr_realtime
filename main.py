
import time
import cv2
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt as math_sqrt
from numpy import array as np_array, copy as np_copy
from joblib import load as load_model
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# 手特征点检测
CONFIDENCE_DETECT = 0.3 # 默认0.5，值越小，越容易认为检测到手
CONFIDENCE_TRACK = 0.3 # 默认0.5，值越小，更偏向于使用跟踪而不是重新检测手
MODEL_PATH = "./model/hand_landmarker.task"
# 手势识别
K = 5 # 模型原本是3
KNN_MODEL_PATH = "./model/knn_slr_model.pkl"
SCALER_PATH = './model/distance_scaler.pkl'
# 绘制
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

class landmarker_and_result():
    def __init__(self):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker()
   
    def createLandmarker(self):
        # callback function
        def update_result(result, output_image, timestamp_ms):
            self.result = result
        # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
        options = mp.tasks.vision.HandLandmarkerOptions( 
            base_options = mp.tasks.BaseOptions(model_asset_path = MODEL_PATH), 
            running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands = 2, #双手
            min_hand_detection_confidence = CONFIDENCE_DETECT,
            min_hand_presence_confidence = CONFIDENCE_TRACK,
            min_tracking_confidence = CONFIDENCE_TRACK,
            result_callback = update_result)
        
        # initialize landmarker
        self.landmarker = self.landmarker.create_from_options(options)
   
    def detect_async(self, frame):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

    def close(self):
        # close landmarker
        self.landmarker.close()

# 绘制手部关键点
def draw_landmarks_on_image(rgb_image, detection_result):
    try:
        if detection_result.hand_landmarks == []:
            return rgb_image
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = np_copy(rgb_image)
            # annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

                # Get the top left corner of the detected hand's bounding box.
                height, width, _ = annotated_image.shape
                x_coordinates = [landmark.x for landmark in hand_landmarks]
                y_coordinates = [landmark.y for landmark in hand_landmarks]
                text_x = int(min(x_coordinates) * width)
                text_y = int(min(y_coordinates) * height)

                prediction, max_probs = classifier(detection_result)

                cv2.putText(annotated_image, f"NUM{prediction},P{max_probs*100}%",
                            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_4)
            return annotated_image
    except Exception as e:
        import traceback
        print("发生异常：", e)        
        print("详细信息：")
        traceback.print_exc()
        return rgb_image


# 使用KNN识别手势
def classifier(result):
    distances = []
    base_point = result.hand_world_landmarks[0][0]  # 取第一只手的第一个关键点作为基准

    # 计算20个距离 
    for landmark in result.hand_world_landmarks[0][1:]:
        distances.append(math_sqrt((base_point.x - landmark.x) ** 2 + (base_point.y - landmark.y) ** 2))

    # 将距离转换为正确的形状 (必须是2D数组)
    distances_array = np_array(distances).reshape(1, -1)
    
    # 加载训练时保存的归一化器
    scaler = load_model(SCALER_PATH)
    
    # 使用相同的归一化器对新数据进行变换
    normalized_data = scaler.transform(distances_array)
 
    # 进行KNN预测
    knn = load_model(KNN_MODEL_PATH)
    knn.n_neighbors = K
    
    prediction = knn.predict(normalized_data)
    # 预测类别概率（置信度）
    probs = knn.predict_proba(normalized_data)
    max_probs = probs.max(axis=1)[0]

    return prediction, max_probs


def main():
    # create landmarker
    hand_landmarker = landmarker_and_result()

    # access webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # pull frame
        ret, frame = cap.read()
        # mirror frame
        frame = cv2.flip(frame, 1)
        # update landmarker results
        hand_landmarker.detect_async(frame)
        # draw landmarks on frame
        frame = draw_landmarks_on_image(frame,hand_landmarker.result)
        # display image
        cv2.imshow('slr (press q to exit)',frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # release everything
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
