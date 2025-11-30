import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

class AdvancedHandScroll:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.scroll_sensitivity = 15
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.1  # seconds
        
    def get_finger_state(self, hand_landmarks):
        """Check which fingers are extended"""
        landmarks = hand_landmarks.landmark
        finger_states = []
        
        # Finger tips and MCP joints
        finger_pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]  # Index, Middle, Ring, Pinky
        
        for tip, mcp in finger_pairs:
            if landmarks[tip].y < landmarks[mcp].y:  # Finger extended
                finger_states.append(True)
            else:
                finger_states.append(False)
        
        # Thumb (different calculation)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        thumb_extended = thumb_tip.x < thumb_mcp.x
        finger_states.append(thumb_extended)
        
        return finger_states  # [Index, Middle, Ring, Pinky, Thumb]
    
    def detect_gesture(self, finger_states):
        """Detect specific gestures"""
        index, middle, ring, pinky, thumb = finger_states
        
        # Scroll Up: Index finger only
        if index and not middle and not ring and not pinky:
            return "SCROLL_UP"
        
        # Scroll Down: Middle finger only
        elif middle and not index and not ring and not pinky:
            return "SCROLL_DOWN"
        
        # Fast Scroll: Index + Middle
        elif index and middle and not ring and not pinky:
            return "FAST_SCROLL"
        
        # Stop: Fist or other gestures
        else:
            return "STOP"
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("🎯 ADVANCED HAND SCROLL CONTROLLER")
        print("📝 Gesture Mapping:")
        print("   👆 Telunjuk saja = Scroll Up")
        print("   ✌️ Tengah saja = Scroll Down") 
        print("   👆✌️ Telunjuk+Tengah = Fast Scroll")
        print("   ✊ Kepal = Stop")
        print("   Press 'Q' to quit")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            results = self.hands.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            current_gesture = "NO HAND"
            scroll_action = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    finger_states = self.get_finger_state(hand_landmarks)
                    gesture = self.detect_gesture(finger_states)
                    current_gesture = gesture
                    
                    current_time = time.time()
                    if current_time - self.last_scroll_time > self.scroll_cooldown:
                        
                        if gesture == "SCROLL_UP":
                            pyautogui.scroll(3 * self.scroll_sensitivity)
                            scroll_action = "🔼 SCROLL UP"
                            self.last_scroll_time = current_time
                            
                        elif gesture == "SCROLL_DOWN":
                            pyautogui.scroll(-3 * self.scroll_sensitivity)
                            scroll_action = "🔽 SCROLL DOWN"
                            self.last_scroll_time = current_time
                            
                        elif gesture == "FAST_SCROLL":
                            pyautogui.scroll(8 * self.scroll_sensitivity)
                            scroll_action = "⚡ FAST SCROLL"
                            self.last_scroll_time = current_time
            
            # Display information
            cv2.putText(image_bgr, f"Gesture: {current_gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image_bgr, f"Action: {scroll_action}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Gesture guide
            cv2.putText(image_bgr, "👆=Up  ✌️=Down  👆✌️=Fast  ✊=Stop", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image_bgr, "Press 'Q' to quit", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Advanced Hand Scroll', image_bgr)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    advanced_scroll = AdvancedHandScroll()
    advanced_scroll.run()