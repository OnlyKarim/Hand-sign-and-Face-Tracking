import cv2
import mediapipe as mp
import numpy as np
import time

class UltimateHandBlock:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0, # Diatur ke 0 agar FPS bisa mencapai 60
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Inisialisasi blok pertama
        self.blocks = [
            {'pos': [540, 260], 'size': 150, 'color': (255, 150, 0), 'id': time.time()}
        ]
        self.last_split_time = 0

    def get_hand_data(self, results, w, h):
        """Mengambil data ujung jari (4, 8, 12, 16, 20) dari kedua tangan"""
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                tips = []
                for tid in [4, 8, 12, 16, 20]:
                    lm = hand_lms.landmark[tid]
                    tips.append((int(lm.x * w), int(lm.y * h)))
                
                # Hitung pusat tangan (rata-rata ujung jari)
                avg_x = int(np.mean([p[0] for p in tips]))
                avg_y = int(np.mean([p[1] for p in tips]))
                
                # Hitung tingkat 'kebukaan' tangan (jarak jempol ke kelingking)
                span = np.hypot(tips[0][0] - tips[4][0], tips[0][1] - tips[4][1])
                
                hands_data.append({
                    'tips': tips,
                    'center': (avg_x, avg_y),
                    'span': span
                })
        return hands_data

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

        p_time = 0

        while cap.isOpened():
            success, img = cap.read()
            if not success: break
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            hands_info = self.get_hand_data(results, w, h)

            # Logika Manipulasi
            if len(hands_info) == 2:
                h1, h2 = hands_info[0], hands_info[1]
                # Jarak antara pusat kedua tangan
                dist_between_hands = np.hypot(h1['center'][0] - h2['center'][0], 
                                              h1['center'][1] - h2['center'][1])
                
                mid_point = ((h1['center'][0] + h2['center'][0]) // 2, 
                             (h1['center'][1] + h2['center'][1]) // 2)

                for block in self.blocks:
                    bx, by = block['pos']
                    bs = block['size']
                    
                    # Cek apakah kedua tangan berada di sekitar blok
                    if (bx - 50 < mid_point[0] < bx + bs + 50 and 
                        by - 50 < mid_point[1] < by + bs + 50):
                        
                        # 1. SCALE: Ukuran blok mengikuti jarak kedua tangan
                        if dist_between_hands > 50:
                            block['size'] = int(dist_between_hands * 0.7)
                            # Update posisi agar tetap di tengah tangan
                            block['pos'] = [mid_point[0] - block['size']//2, 
                                            mid_y := mid_point[1] - block['size']//2]

                        # 2. SPLIT: Jika tangan merapat lalu tiba-tiba menjauh sangat cepat
                        if dist_between_hands > 500 and (time.time() - self.last_split_time) > 1.5:
                            new_block = block.copy()
                            new_block['id'] = time.time()
                            new_block['pos'] = [h2['center'][0], h2['center'][1]]
                            self.blocks.append(new_block)
                            self.last_split_time = time.time()
                            break
            
            elif len(hands_info) == 1:
                # Drag sederhana dengan 1 tangan (5 jari merapat)
                h1 = hands_info[0]
                for block in self.blocks:
                    bx, by = block['pos']
                    bs = block['size']
                    if bx < h1['center'][0] < bx + bs and by < h1['center'][1] < by + bs:
                        block['pos'] = [h1['center'][0] - bs//2, h1['center'][1] - bs//2]

            # Render Blok
            for block in self.blocks:
                cv2.rectangle(img, (block['pos'][0], block['pos'][1]), 
                              (block['pos'][0] + block['size'], block['pos'][1] + block['size']), 
                              block['color'], -1)
                cv2.rectangle(img, (block['pos'][0], block['pos'][1]), 
                              (block['pos'][0] + block['size'], block['pos'][1] + block['size']), 
                              (255, 255, 255), 3)

            # Render 10 Jari
            for hand in hands_info:
                for tip in hand['tips']:
                    cv2.circle(img, tip, 10, (0, 255, 255), cv2.FILLED)

            # FPS & UI
            c_time = time.time()
            fps = 1/(c_time-p_time) if c_time-p_time > 0 else 0
            p_time = c_time
            cv2.putText(img, f"FPS: {int(fps)} | Blocks: {len(self.blocks)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "Gunakan 2 TANGAN: Rapatkan/Renggangkan untuk SCALE | Tarik JAUH untuk SPLIT", 
                        (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("10-Finger Master Manipulator", img)
            if cv2.waitKey(1) & 0xFF == 27: break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    UltimateHandBlock().run()