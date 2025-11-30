import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

class HandScrollCursor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Scroll parameters
        self.scroll_sensitivity = 15
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.1  # seconds
        
        # Cursor parameters
        self.cursor_active = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.cursor_smoothing = 0.7
        self.last_cursor_pos = None
        
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
        
        # Cursor Control: Index + Thumb extended (OK gesture)
        if index and thumb and not middle and not ring and not pinky:
            return "CURSOR"
        
        # Scroll Up: Index finger only
        elif index and not middle and not ring and not pinky and not thumb:
            return "SCROLL_UP"
        
        # Scroll Down: Middle finger only
        elif middle and not index and not ring and not pinky and not thumb:
            return "SCROLL_DOWN"
        
        # Fast Scroll: Index + Middle
        elif index and middle and not ring and not pinky and not thumb:
            return "FAST_SCROLL"
        
        # Click: Index + Middle + Thumb
        elif index and middle and thumb and not ring and not pinky:
            return "CLICK"
        
        # Right Click: All fingers extended
        elif index and middle and ring and pinky and thumb:
            return "RIGHT_CLICK"
        
        # Stop: Fist or other gestures
        else:
            return "STOP"
    
    def move_cursor(self, hand_landmarks, image_width, image_height):
        """Move cursor based on hand position"""
        landmarks = hand_landmarks.landmark
        
        # Use index finger tip for cursor position
        index_tip = landmarks[8]
        
        # Convert normalized coordinates to screen coordinates
        cursor_x = int(index_tip.x * self.screen_width)
        cursor_y = int(index_tip.y * self.screen_height)
        
        # Smooth cursor movement
        if self.last_cursor_pos:
            smooth_x = int(self.cursor_smoothing * cursor_x + (1 - self.cursor_smoothing) * self.last_cursor_pos[0])
            smooth_y = int(self.cursor_smoothing * cursor_y + (1 - self.cursor_smoothing) * self.last_cursor_pos[1])
        else:
            smooth_x, smooth_y = cursor_x, cursor_y
        
        # Move cursor
        pyautogui.moveTo(smooth_x, smooth_y, duration=0.1)
        self.last_cursor_pos = (smooth_x, smooth_y)
        
        return cursor_x, cursor_y
    
    def draw_cursor_info(self, image, cursor_x, cursor_y, gesture):
        """Draw cursor information on image"""
        # Draw crosshair at cursor position
        crosshair_size = 20
        cv2.line(image, (cursor_x - crosshair_size, cursor_y), 
                (cursor_x + crosshair_size, cursor_y), (0, 255, 0), 2)
        cv2.line(image, (cursor_x, cursor_y - crosshair_size), 
                (cursor_x, cursor_y + crosshair_size), (0, 255, 0), 2)
        
        # Draw circle around cursor
        cv2.circle(image, (cursor_x, cursor_y), 30, (0, 255, 0), 2)
        
        # Display cursor coordinates
        coord_text = f"Cursor: ({cursor_x}, {cursor_y})"
        cv2.putText(image, coord_text, (cursor_x + 40, cursor_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display click status
        if gesture == "CLICK":
            cv2.putText(image, "CLICKING", (cursor_x + 40, cursor_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif gesture == "RIGHT_CLICK":
            cv2.putText(image, "RIGHT CLICK", (cursor_x + 40, cursor_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("🎯 ADVANCED HAND SCROLL + CURSOR CONTROLLER")
        print("📝 Gesture Mapping:")
        print("   👆 Telunjuk saja = Scroll Up")
        print("   ✌️ Tengah saja = Scroll Down") 
        print("   👆✌️ Telunjuk+Tengah = Fast Scroll")
        print("   👌 Telunjuk+Ibu jari = Kursor Mouse")
        print("   👆✌️👍 Telunjuk+Tengah+Ibu jari = Klik")
        print("   🖐️ Semua jari terbuka = Klik Kanan")
        print("   ✊ Kepal = Stop")
        print("   Press 'Q' to quit")
        
        cursor_x, cursor_y = 0, 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            image = cv2.flip(image, 1)
            image_height, image_width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            results = self.hands.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            current_gesture = "NO HAND"
            scroll_action = ""
            cursor_action = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    finger_states = self.get_finger_state(hand_landmarks)
                    gesture = self.detect_gesture(finger_states)
                    current_gesture = gesture
                    
                    current_time = time.time()
                    
                    # Handle different gestures
                    if gesture == "CURSOR":
                        cursor_x, cursor_y = self.move_cursor(hand_landmarks, image_width, image_height)
                        cursor_action = "🖱️ CURSOR MOVING"
                        self.draw_cursor_info(image_bgr, cursor_x, cursor_y, gesture)
                        
                    elif gesture == "CLICK":
                        # Move cursor first
                        cursor_x, cursor_y = self.move_cursor(hand_landmarks, image_width, image_height)
                        # Then click
                        pyautogui.click()
                        cursor_action = "🖱️ CLICK"
                        self.draw_cursor_info(image_bgr, cursor_x, cursor_y, gesture)
                        time.sleep(0.3)  # Prevent multiple clicks
                        
                    elif gesture == "RIGHT_CLICK":
                        # Move cursor first
                        cursor_x, cursor_y = self.move_cursor(hand_landmarks, image_width, image_height)
                        # Then right click
                        pyautogui.rightClick()
                        cursor_action = "🖱️ RIGHT CLICK"
                        self.draw_cursor_info(image_bgr, cursor_x, cursor_y, gesture)
                        time.sleep(0.3)  # Prevent multiple clicks
                        
                    elif gesture == "SCROLL_UP" and current_time - self.last_scroll_time > self.scroll_cooldown:
                        pyautogui.scroll(3 * self.scroll_sensitivity)
                        scroll_action = "🔼 SCROLL UP"
                        self.last_scroll_time = current_time
                        
                    elif gesture == "SCROLL_DOWN" and current_time - self.last_scroll_time > self.scroll_cooldown:
                        pyautogui.scroll(-3 * self.scroll_sensitivity)
                        scroll_action = "🔽 SCROLL DOWN"
                        self.last_scroll_time = current_time
                        
                    elif gesture == "FAST_SCROLL" and current_time - self.last_scroll_time > self.scroll_cooldown:
                        pyautogui.scroll(8 * self.scroll_sensitivity)
                        scroll_action = "⚡ FAST SCROLL"
                        self.last_scroll_time = current_time
            
            # Display information panel
            cv2.rectangle(image_bgr, (5, 5), (400, 160), (0, 0, 0), -1)
            cv2.rectangle(image_bgr, (5, 5), (400, 160), (255, 255, 255), 2)
            
            cv2.putText(image_bgr, f"Gesture: {current_gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image_bgr, f"Scroll: {scroll_action}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image_bgr, f"Cursor: {cursor_action}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Gesture guide
            cv2.putText(image_bgr, "👆=Up  ✌️=Down  👆✌️=Fast  👌=Cursor", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(image_bgr, "👆✌️👍=Click  🖐️=RightClick  ✊=Stop", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(image_bgr, "Press 'Q' to quit", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('Hand Scroll + Cursor Control', image_bgr)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_controller = HandScrollCursor()
    hand_controller.run()