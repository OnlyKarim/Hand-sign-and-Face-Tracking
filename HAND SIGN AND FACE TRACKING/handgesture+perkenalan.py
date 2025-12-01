import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os
import pygame  # Untuk memutar audio
from gtts import gTTS  # Google Text-to-Speech

class BISINDOIntroductionRecognizer:
    def __init__(self):
        # Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Inisialisasi pygame untuk audio
        pygame.mixer.init()
        
        # State variables
        self.current_state = "IDLE"
        self.state_start_time = 0
        self.sequence = []
        self.gesture_hold_time = 1.5
        self.is_speaking = False
        
        # Text untuk display
        self.display_text = "Mulai dengan gesture: TANGAN TERBUKA"
        
        # Warna untuk tiap state
        self.state_colors = {
            "HALO": (0, 255, 0),       # Hijau
            "NAMA": (255, 255, 0),     # Kuning
            "JEMPOL": (0, 255, 255),   # Cyan
            "KELINGKING": (255, 0, 255), # Magenta
            "METAL": (255, 165, 0),    # Orange
            "IDLE": (200, 200, 200)    # Abu-abu
        }
        
        # Mapping gesture ke suara (dalam bahasa Indonesia)
        self.gesture_sounds = {
            "HALO": "Halo perkenalkan",
            "NAMA": "Nama saya",
            "JEMPOL": "Hafizh Karim Fauzi",
            "KELINGKING": "Saya berasal dari Teknik Komputer Institut Teknologi Sepuluh Nopember",
            "METAL": "Salam kenal"
        }
        
        # Buat folder untuk menyimpan file audio
        if not os.path.exists("audio_cache"):
            os.makedirs("audio_cache")
        
        # Pre-generate audio files untuk performa lebih baik
        print("Menyiapkan suara...")
        self.prepare_audio_files()
    
    def prepare_audio_files(self):
        """Buat file audio terlebih dahulu untuk menghindari delay"""
        for gesture, text in self.gesture_sounds.items():
            filename = f"audio_cache/{gesture}.mp3"
            if not os.path.exists(filename):
                try:
                    tts = gTTS(text=text, lang='id', slow=False)
                    tts.save(filename)
                    print(f"✅ Suara untuk '{gesture}' siap")
                except Exception as e:
                    print(f"⚠️  Gagal membuat suara untuk '{gesture}': {e}")
    
    def speak_with_gtts(self, text):
        """Menggunakan Google TTS dengan threading"""
        def speak_thread():
            try:
                filename = f"audio_cache/temp_{int(time.time())}.mp3"
                
                # Buat audio dengan gTTS
                tts = gTTS(text=text, lang='id', slow=False)
                tts.save(filename)
                
                # Putar audio dengan pygame
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                
                # Tunggu sampai selesai
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Hapus file sementara
                try:
                    os.remove(filename)
                except:
                    pass
                    
            except Exception as e:
                print(f"Error dalam TTS: {e}")
            finally:
                self.is_speaking = False
        
        # Cek apakah sudah ada suara yang sedang diputar
        if not self.is_speaking:
            self.is_speaking = True
            thread = threading.Thread(target=speak_thread)
            thread.daemon = True
            thread.start()
    
    def speak_prepared_audio(self, gesture):
        """Menggunakan audio yang sudah dipersiapkan sebelumnya"""
        def play_audio():
            try:
                filename = f"audio_cache/{gesture}.mp3"
                if os.path.exists(filename):
                    pygame.mixer.music.load(filename)
                    pygame.mixer.music.play()
                    
                    # Tunggu sampai selesai
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                else:
                    # Fallback ke gTTS realtime
                    self.speak_with_gtts(self.gesture_sounds[gesture])
                    
            except Exception as e:
                print(f"Error memutar audio: {e}")
            finally:
                self.is_speaking = False
        
        if not self.is_speaking:
            self.is_speaking = True
            thread = threading.Thread(target=play_audio)
            thread.daemon = True
            thread.start()
    
    def get_finger_states(self, landmarks):
        """Mendeteksi apakah jari-jari terbuka atau tertutup"""
        finger_states = {
            'thumb': False,
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [2, 6, 10, 14, 18]
        
        for i in range(5):
            if landmarks[tip_ids[i]].y < landmarks[pip_ids[i]].y:
                if i == 0:
                    if landmarks[tip_ids[i]].x < landmarks[pip_ids[i]].x:
                        finger_states[list(finger_states.keys())[i]] = True
                else:
                    finger_states[list(finger_states.keys())[i]] = True
        
        return finger_states
    
    def detect_gesture(self, landmarks):
        """Deteksi gesture berdasarkan state jari-jari"""
        finger_states = self.get_finger_states(landmarks)
        
        thumb = finger_states['thumb']
        index = finger_states['index']
        middle = finger_states['middle']
        ring = finger_states['ring']
        pinky = finger_states['pinky']
        
        # Rule-based gesture detection
        if all([thumb, index, middle, ring, pinky]):
            return "HALO"
        elif index and not middle and not ring and not pinky:
            return "NAMA"
        elif thumb and not index and not middle and not ring and not pinky:
            return "JEMPOL"
        elif pinky and not index and not middle and not ring:
            return "KELINGKING"
        elif index and pinky and not middle and not ring:
            return "METAL"
        
        return "IDLE"
    
    def update_state(self, gesture):
        """State machine untuk sequence perkenalan"""
        current_time = time.time()
        
        # Reset timer jika gesture berubah
        if gesture != self.current_state:
            self.current_state = gesture
            self.state_start_time = current_time
            return False
        
        # Konfirmasi gesture jika dipertahankan > hold_time
        elif current_time - self.state_start_time > self.gesture_hold_time:
            if gesture == "HALO" and "HALO" not in self.sequence:
                self.sequence.append("HALO")
                self.display_text = "HALO PERKENALAN\nSelanjutnya: TELUNJUK"
                self.speak_prepared_audio("HALO")
                return True
            elif gesture == "NAMA" and "NAMA" not in self.sequence and "HALO" in self.sequence:
                self.sequence.append("NAMA")
                self.display_text = "NAMA SAYA\nSelanjutnya: JEMPOL"
                self.speak_prepared_audio("NAMA")
                return True
            elif gesture == "JEMPOL" and "JEMPOL" not in self.sequence and "NAMA" in self.sequence:
                self.sequence.append("JEMPOL")
                self.display_text = "HAFIZH KARIM FAUZI\nSelanjutnya: KELINGKING"
                self.speak_prepared_audio("JEMPOL")
                return True
            elif gesture == "KELINGKING" and "KELINGKING" not in self.sequence and "JEMPOL" in self.sequence:
                self.sequence.append("KELINGKING")
                self.display_text = "TEKNIK KOMPUTER ITS\nSelanjutnya: METAL"
                self.speak_prepared_audio("KELINGKING")
                return True
            elif gesture == "METAL" and "METAL" not in self.sequence and "KELINGKING" in self.sequence:
                self.sequence.append("METAL")
                self.display_text = "SALAM KENAL\nPERKENALAN SELESAI"
                self.speak_prepared_audio("METAL")
                return True
        
        return False
    
    def process_frame(self, frame):
        """Proses frame dan deteksi gesture"""
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Gambar UI background untuk text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-200), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
                # Detect gesture
                gesture = self.detect_gesture(hand_landmarks.landmark)
                
                # Update state
                self.update_state(gesture)
                
                # Get wrist position
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                
                # Tampilkan gesture saat ini
                cv2.putText(frame, f"Gesture: {gesture}", 
                           (wrist_x - 50, wrist_y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           self.state_colors.get(gesture, (255, 255, 255)), 2)
                
                # Gambar progress circle
                if self.current_state != "IDLE":
                    elapsed = time.time() - self.state_start_time
                    progress = min(elapsed / self.gesture_hold_time, 1.0)
                    
                    center = (wrist_x, wrist_y + 50)
                    radius = 30
                    
                    # Outer circle
                    cv2.circle(frame, center, radius, (100, 100, 100), 3)
                    
                    # Progress fill
                    if progress > 0:
                        color = self.state_colors.get(gesture, (255, 255, 255))
                        cv2.circle(frame, center, int(radius * progress), color, -1)
        
        # Tampilkan sequence progress
        y_offset = h - 180
        for i, gesture in enumerate(self.sequence):
            color = self.state_colors.get(gesture, (255, 255, 255))
            cv2.putText(frame, f"{i+1}. {gesture}", 
                       (20, y_offset + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Tampilkan instruksi
        cv2.putText(frame, self.display_text, 
                   (20, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tampilkan jumlah gesture yang sudah dideteksi
        cv2.putText(frame, f"Progress: {len(self.sequence)}/5", 
                   (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Reset sequence jika sudah selesai
        if len(self.sequence) >= 5:
            cv2.putText(frame, "SELESAI Tekan R untuk reset", 
                       (w//2 - 200, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        return frame, self.sequence
    
    def reset_sequence(self):
        """Reset semua state"""
        self.current_state = "IDLE"
        self.sequence = []
        self.display_text = "Mulai dengan gesture: TANGAN TERBUKA"
        self.state_start_time = 0

def main():
    recognizer = BISINDOIntroductionRecognizer()
    cap = cv2.VideoCapture(0)
    
    # Atur FPS tinggi untuk responsif
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("=" * 60)
    print("BISINDO INTRODUCTION WITH GOOGLE TTS")
    print("=" * 60)
    print("Urutan gesture:")
    print("1. ✋ Tangan terbuka      → HALO PERKENALAN")
    print("2. 👆 Telunjuk saja      → NAMA SAYA")
    print("3. 👍 Jempol saja        → Hafizh Karim Fauzi")
    print("4. 🤙 Kelingking saja    → Saya dari Teknik Komputer ITS")
    print("5. 🤘 Metal (Index+Pinky)→ SALAM KENAL")
    print("\nInstruksi:")
    print("- Tahan setiap gesture selama 1.5 detik")
    print("- Sistem akan berbicara dengan suara Google")
    print("- Tekan R untuk reset sequence")
    print("- Tekan Q untuk keluar")
    print("=" * 60)
    
    fps_counter = 0
    fps_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal membaca frame")
            break
        
        # Process frame
        processed_frame, sequence = recognizer.process_frame(frame)
        
        # Hitung dan tampilkan FPS
        fps_counter += 1
        if time.time() - fps_time > 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_time = time.time()
            cv2.putText(processed_frame, f"FPS: {current_fps}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        # Tampilkan frame
        cv2.imshow('BISINDO: Introduction with Google TTS', processed_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.reset_sequence()
            print("Sequence direset")
        elif key == ord('s'):  # Test suara
            print("Testing suara...")
            recognizer.speak_with_gtts("Testing suara dari Google Text to Speech")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Install packages yang diperlukan
    print("Memeriksa dependensi...")
    
    # Cek apakah paket sudah terinstall
    try:
        import pygame
        from gtts import gTTS
        print("✅ Semua paket sudah terinstall")
    except ImportError as e:
        print(f"⚠️  Paket belum terinstall: {e}")
        print("\nSilakan install dengan perintah:")
        print("pip install gtts pygame")
        exit()
    
    # Cek kamera
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("❌ ERROR: Kamera tidak terdeteksi!")
        print("Pastikan kamera terhubung dan tidak digunakan aplikasi lain")
        exit()
    test_cap.release()
    
    # Jalankan program utama
    main()