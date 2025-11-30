import cv2
import mediapipe as mp
import math

class CombinedTracker:
    def __init__(self):
        # Inisialisasi MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def calculate_finger_angles(self, hand_landmarks):
        """Hitung sudut jari"""
        landmarks = hand_landmarks.landmark
        finger_tips = [4, 8, 12, 16, 20]
        finger_mcp = [2, 5, 9, 13, 17]
        
        angles = []
        for i in range(len(finger_tips)):
            tip = landmarks[finger_tips[i]]
            mcp = landmarks[finger_mcp[i]]
            wrist = landmarks[0]
            
            vec1 = [mcp.x - wrist.x, mcp.y - wrist.y]
            vec2 = [tip.x - mcp.x, tip.y - mcp.y]
            
            dot_product = vec1[0]*vec2[0] + vec1[1]*vec2[1]
            mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
            mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
            
            if mag1 * mag2 == 0:
                angle = 0
            else:
                angle = math.acos(dot_product / (mag1 * mag2)) * 180 / math.pi
            
            angles.append(angle)
        
        return angles
    
    def recognize_gesture(self, angles):
        """Recognize hand gesture"""
        thumb, index, middle, ring, pinky = angles
        
        if all(angle > 120 for angle in [index, middle, ring, pinky]):
            return "OPEN HAND"
        elif all(angle < 90 for angle in [index, middle, ring, pinky]):
            return "FIST"
        elif index > 120 and all(angle < 90 for angle in [middle, ring, pinky]):
            return "POINTING"
        elif index > 120 and middle > 120 and all(angle < 90 for angle in [ring, pinky]):
            return "VICTORY"
        elif thumb < 60 and index < 60:
            return "OK"
        else:
            return "UNKNOWN"
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh, \
            self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue
                
                # Konversi BGR ke RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Proses deteksi wajah dan tangan
                face_results = face_mesh.process(image_rgb)
                hand_results = hands.process(image_rgb)
                
                # Konversi kembali ke BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Variabel untuk status
                face_status = "No face detected"
                hand_status = "No hand detected"
                
                # Gambar landmarks wajah
                if face_results.multi_face_landmarks:
                    face_status = "Face detected"
                    for face_landmarks in face_results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                
                # Gambar landmarks tangan dan kenali gesture
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        
                        angles = self.calculate_finger_angles(hand_landmarks)
                        hand_status = self.recognize_gesture(angles)
                
                # Tampilkan status
                cv2.putText(image, f'Face: {face_status}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image, f'Hand: {hand_status}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Tampilkan hasil
                cv2.imshow('Combined Face & Hand Tracking', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        
        cap.release()
        cv2.destroyAllWindows()

# Jalankan combined tracker
if __name__ == "__main__":
    tracker = CombinedTracker()
    tracker.run()