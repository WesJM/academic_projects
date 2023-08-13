import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import EventDetector
import numpy as np
from numpy.linalg import norm
import torch.nn.functional as F
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from google.colab.patches import cv2_imshow

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}

# calculate angle between two n-d unit vectors in degrees
def calc_angle(x, y):
    return 180*np.arccos(np.dot(x,y)/(norm(x)*norm(y)))/np.pi

# calculate angle between two 2D vectors in degrees
# https://github.com/Pradnya1208/Squats-angle-detection-using-OpenCV-and-mediapipe_v1
def calculate_angle(a, b, c):
    a = np.array(a) # first
    b = np.array(b) # mid
    c = np.array(c) # end
    
    # get radians, where [0] = x and [1] = y
    radians = np.arctan2(c[1]- b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


# class SampleVideo(Dataset):
#     def __init__(self, path, input_size=160, transform=None):
#         self.path = path
#         self.input_size = input_size
#         self.transform = transform

#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):
#         cap = cv2.VideoCapture(self.path)
#         frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
#         ratio = self.input_size / max(frame_size)
#         new_size = tuple([int(x * ratio) for x in frame_size])
#         delta_w = self.input_size - new_size[1]
#         delta_h = self.input_size - new_size[0]
#         top, bottom = delta_h // 2, delta_h - (delta_h // 2)
#         left, right = delta_w // 2, delta_w - (delta_w // 2)

#         # preprocess and return frames
#         images = []
#         for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
#             _, img = cap.read()
#             resized = cv2.resize(img, (new_size[1], new_size[0]))
#             b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
#                                        value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

#             b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
#             images.append(b_img_rgb)
#         cap.release()
#         labels = np.zeros(len(images)) # only for compatibility with transforms
#         sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample


class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        
        ##### Resize image -----
        
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return frames
        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)
        
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            
            images.append(b_img_rgb)
        cap.release()
        
        
        ##### Optical Flow image -----
        
        base_frame = images[0]
        
        # Converts frame to grayscale
        previous_frame = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
        
        # Define HSV (hue, saturation, value) color array
        hsv = np.zeros_like(base_frame)
        # Update the color array second dimension to 'white'
        hsv[..., 1] = 255
        
        # Define parameters for Gunnar Farneback algorithm
        feature_params = dict(pyr_scale=0.5,
                              levels=3,
                              winsize=15,
                              iterations=3,
                              poly_n=5,
                              poly_sigma=1.2,
                              flags=0)
        
        optical_images = []
        # Iterate video frames
        for i in range(len(images)):
            frame = images[i]
        
            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # Define an optical flow object
            flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, None, **feature_params)
        
            # Calculate the magnitude and angle the vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
            # Sets image hue (in HSV array) to the optical flow direction
            hsv[..., 0] = angle * 180 / np.pi / 2
        
            # Set image value (in HSV array) to normalized magnitude
            # Cleaner output
            clean_hsv = hsv.copy()
            clean_hsv[..., 2] = np.minimum(45 * magnitude, 255)
        
            # Convert HSV to RGB (BGR) representation
            clean_rgb = cv2.cvtColor(clean_hsv, cv2.COLOR_HSV2BGR)
        
            # Write video
            optical_images.append(clean_rgb)
        
            # Update the previous frame
            previous_frame = next_frame
        
        
        ##### Preprocess Image -----
        
        frame_size = [optical_images[i].shape[0], optical_images[i].shape[1]]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
            
        # preprocess and return frames
        images = []
        for i in range(len(optical_images)):
            img = optical_images[i]
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)
        
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            
            images.append(b_img_rgb)
        
        labels = np.zeros(len(images)) # only for compatibility with transforms
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    ## initialize pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, smooth_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
    custom_connections = list(mp_pose.POSE_CONNECTIONS)
    
    # list of landmarks to exclude from the drawing
    excluded_landmarks = [
        mp_pose.PoseLandmark.LEFT_EYE, 
        mp_pose.PoseLandmark.RIGHT_EYE, 
        mp_pose.PoseLandmark.LEFT_EYE_INNER, 
        mp_pose.PoseLandmark.RIGHT_EYE_INNER, 
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.MOUTH_LEFT,
        mp_pose.PoseLandmark.MOUTH_RIGHT ]

    for landmark in excluded_landmarks:
        # change the way the excluded landmarks are drawn
        custom_style[landmark] = DrawingSpec(color=(255,255,0), thickness=None) 

        # remove all connections which contain these landmarks
        custom_connections = [connection_tuple for connection_tuple in custom_connections 
                                if landmark.value not in connection_tuple]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default='test_video.mp4')
    parser.add_argument('-p2', '--path2', help='Path to video that you want to compare', default='test_video.mp4')
    parser.add_argument('-a', '--angle', help='Angle from which video is taken (open or behind)', default='open')
    parser.add_argument('-a2', '--angle2', help='Angle from which video is taken (open or behind)', default='open')
    parser.add_argument('-rl', '--hand', help='Handedness of swing', default='right')
    parser.add_argument('-rl2', '--hand2', help='Handedness of swing', default='right')
    parser.add_argument('-f', '--feature', help='Biomechanical feature of interest', default='relbow_angle')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
    args = parser.parse_args()
    seq_length = args.seq_length
    
    if args.path2 == 'testing_video.mp4':
        video_paths = [args.path]
    else:
        video_paths = [args.path, args.path2]
        
    videos = [] # array to keep array of frames (which have features for each frame) for all videos
    
    for path in video_paths:
        if path == args.path2:
            hand = args.hand2
        else:
            hand = args.hand

        print('Preparing video: {}'.format(path))

        ds = SampleVideo(path, transform=transforms.Compose([ToTensor(),
                                    Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])]))

        dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

        model = EventDetector(pretrain=True,
                            width_mult=1.,
                            lstm_layers=1,
                            lstm_hidden=256,
                            bidirectional=True,
                            dropout=False)

        try:
            save_dict = torch.load('models/swingnet_2000.pth.tar')
        except:
            print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        model.load_state_dict(save_dict['model_state_dict'])
        model.to(device)
        model.eval()
        print("Loaded model weights")

        print('Testing...')
        for sample in dl:
            images = sample['images']
            # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
            batch = 0
            while batch * seq_length < images.shape[1]:
                if (batch + 1) * seq_length > images.shape[1]:
                    image_batch = images[:, batch * seq_length:, :, :, :]
                else:
                    image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                logits = model(image_batch.cuda())
                if batch == 0:
                    probs = F.softmax(logits.data, dim=1).cpu().numpy()
                else:
                    probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
                batch += 1

        events = np.argmax(probs, axis=0)[:-1]
        print('Predicted event frames: {}'.format(events))
        cap = cv2.VideoCapture(path)

        confidence = []
        for i, e in enumerate(events):
            confidence.append(probs[e, i])
        print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))
        
        
        video_frames = []
        for i, e in enumerate(events):
            cap.set(cv2.CAP_PROP_POS_FRAMES, e)
            _, img = cap.read()
            # cv2.putText(img, '{:.3f}'.format(confidence[i]), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
            # cv2_imshow(img)
            
            # convert to RGB
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # process the frame for pose detection
            pose_results = pose.process(frame_rgb)
            
            # body parts location
            rshoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            lshoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            
            rhip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            lhip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            
            relbow = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            lelbow = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            
            rwrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            lwrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            
            rfinger = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
            lfinger = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
            
            rankle = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            lankle = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            
            rknee = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            lknee = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            
            rfoot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            lfoot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            
            
            # # Biomechanical Features
            video_features = {}
            
            video_features['frame_number'] = e
            video_features['frame'] = event_names[i]
            
            
            # bilateral components
            video_features['shoulder_tilt'] = calculate_angle([rshoulder.x, rshoulder.y],[lshoulder.x, lshoulder.y],[rshoulder.x, lshoulder.y])
            video_features['hip_tilt'] = calculate_angle([rhip.x, rhip.y],[lhip.x, lhip.y],[rhip.x, lhip.y])
            video_features['spine_angle'] = calculate_angle([(rshoulder.x + lshoulder.x)/2, (rshoulder.y + lshoulder.y)/2],[(rhip.x + lhip.x)/2, (rhip.y + lhip.y)/2],[(rshoulder.x + lshoulder.x)/2, (rhip.y + lhip.y)/2])
            video_features['hip_translation'] = (rhip.x + lhip.x)/2 - (rfoot.x + lfoot.x)/2
            video_features['shoulder_rotation'] = np.sqrt((lshoulder.x - rshoulder.x)**2+(lshoulder.y - rshoulder.y)**2) # shoulder width for measuring shoulder rotation - behind = more is more rotation, open = less is more rotation
            video_features['hip_rotation'] = np.sqrt((lhip.x - rhip.x)**2+(lhip.x - rhip.x)**2)
            video_features['hand_height'] = rwrist.y - (rshoulder.y + lshoulder.y)/2
            video_features['hand_distance'] = rwrist.x - (rshoulder.x + lshoulder.x)/2
            # club angle?
            
            # right side
            video_features['rhip_angle'] = calculate_angle([rshoulder.x, rshoulder.y], [rhip.x, rhip.y], [rknee.x, rknee.y])
            video_features['rankle_angle'] = calculate_angle([rknee.x, rknee.y],[rankle.x, rankle.y],[rfoot.x, rfoot.y])
            video_features['rknee_angle'] = calculate_angle([rhip.x, rhip.y],[rknee.x, rknee.y],[rankle.x, rankle.y])
            video_features['relbow_angle'] = calculate_angle([rshoulder.x, rshoulder.y],[relbow.x, relbow.y],[rwrist.x, rwrist.y])
            video_features['rwrist_angle'] = calculate_angle([relbow.x, relbow.y],[rwrist.x, rwrist.y],[rfinger.x, rfinger.y])
            video_features['rknee_toe_translation'] = rknee.x - rfoot.x
            
            # left side
            video_features['lhip_angle'] = calculate_angle([lshoulder.x, lshoulder.y], [lhip.x, lhip.y], [lknee.x, lknee.y])
            video_features['lankle_angle'] = calculate_angle([lknee.x, lknee.y],[lankle.x, lankle.y],[lfoot.x, lfoot.y])
            video_features['lknee_angle'] = calculate_angle([lhip.x, lhip.y],[lknee.x, lknee.y],[lankle.x, lankle.y])
            video_features['lelbow_angle'] = calculate_angle([lshoulder.x, lshoulder.y],[lelbow.x, lelbow.y],[lwrist.x, lwrist.y])
            video_features['lwrist_angle'] = calculate_angle([lelbow.x, lelbow.y],[lwrist.x, lwrist.y],[lfinger.x, lfinger.y])
            video_features['lknee_toe_translation'] = lknee.x - lfoot.x
            
            video_frames.append(video_features)
            
            
            cv2.putText(img, '{:.3f}'.format(video_features['relbow_angle']),(20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
            # draw skeleton on the frame
            mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # display the frame
            cv2_imshow(img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        videos.append(video_frames)

    if len(video_paths) == 2:
        features = []
        for feature in videos[0][0]:
            features.append(feature)
            
        # starting widths to compare with frame by frame to determine rotation
        start_shoulder_width1 = videos[0][0]['shoulder_rotation']
        start_shoulder_width2 = videos[1][0]['shoulder_rotation']
        start_hip_width1 = videos[0][0]['hip_rotation']
        start_hip_width2 = videos[1][0]['hip_rotation']

        
        analysis = {} # frame by frame comparitive analysis
        
        # if backswing tempo > 1, then time between toe-up and mid-backswing is less than mid-backswing to top
        backswing_tempo1 = (videos[0][3]['frame_number']-videos[0][2]['frame_number'])/(videos[0][2]['frame_number']-videos[0][1]['frame_number'])
        backswing_tempo2 = (videos[1][3]['frame_number']-videos[1][2]['frame_number'])/(videos[1][2]['frame_number']-videos[1][1]['frame_number'])
        analysis['backswing_tempo'] = [backswing_tempo1, backswing_tempo2]
        
        for feature in features:
            feature_frame = []
            for i in range(len(videos[0])):
                video1_features = videos[0][i]
                video2_features = videos[1][i]
                if feature == 'shoulder_rotation':
                    # 1 = no rotation, higher = more rotation
                    if args.angle == 'behind':
                        shoulder_ratio1 = video1_features[feature]/start_shoulder_width1 
                    else:
                        shoulder_ratio1 = start_shoulder_width1/video1_features[feature]
                    if args.angle2 == 'behind':
                        shoulder_ratio2 = video2_features[feature]/start_shoulder_width2
                    else:
                        shoulder_ratio2 = start_shoulder_width2/video2_features[feature]
                    comparison = [shoulder_ratio1, shoulder_ratio2]
                elif feature == 'hip_rotation':
                    # 1 = no rotation, higher = more rotation
                    if args.angle == 'behind':
                        hip_ratio1 = video1_features[feature]/start_hip_width1
                    else:
                        hip_ratio1 = start_hip_width1/video1_features[feature]
                    if args.angle2 == 'behind':
                        hip_ratio2 = video2_features[feature]/start_hip_width2
                    else:
                        hip_ratio2 = start_hip_width2/video2_features[feature]
                    comparison = [hip_ratio1, hip_ratio2]
                else:
                    comparison = [video1_features[feature], video2_features[feature]]

                if feature != 'shoulder_rotation' and feature != 'hip_rotation':
                    if not isinstance(comparison[0], str):
                        comparison = [round(comparison[0]), round(comparison[1])]
                else:
                    comparison = [round(comparison[0], 2), round(comparison[1], 2)]
                feature_frame.append(comparison)
            
            analysis[feature] = feature_frame
    
    else:
        features = []
        for feature in videos[0][0]:
            features.append(feature)
            
        start_shoulder_width = videos[0][0]['shoulder_rotation']
        start_hip_width = videos[0][0]['hip_rotation']
        
        analysis = {} # frame by frame comparitive analysis
        
        # if backswing tempo > 1, then time between toe-up and mid-backswing is less than mid-backswing to top
        backswing_tempo = (videos[0][3]['frame_number']-videos[0][2]['frame_number'])/(videos[0][2]['frame_number']-videos[0][1]['frame_number'])
        analysis['backswing_tempo'] = backswing_tempo
        
        for feature in features:
            feature_frame = []
            for i in range(len(videos[0])):
                video_features = videos[0][i]
                
                if feature == 'shoulder_rotation':
                    shoulder_ratio = video_features[feature]/start_shoulder_width
                    feature_value = shoulder_ratio
                elif feature == 'hip_rotation':
                    hip_ratio = video_features[feature]/start_shoulder_width
                    feature_value = hip_ratio
                else:
                    feature_value = video_features[feature]
                
                if feature != 'shoulder_rotation' and feature != 'hip_rotation':
                    if not isinstance(feature_value, str):
                        feature_value = round(feature_value)
                else:
                    feature_value = round(feature_value, 2)
                feature_frame.append(feature_value)
            analysis[feature] = feature_frame
        
    
    print(args.feature)
    for frame in range(len((analysis[args.feature]))):
        first = analysis[args.feature][frame][0]
        second = analysis[args.feature][frame][1]
        print(event_names[frame] + ':', analysis[args.feature][frame], '-> ' + str(round(100*(second-first)/first)) + '% difference')