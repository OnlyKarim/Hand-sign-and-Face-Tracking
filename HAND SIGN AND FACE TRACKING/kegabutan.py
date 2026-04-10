import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class SpatialAutoCube:
    def __init__(self):
        # 1. Inisialisasi MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        # Model complexity 0 untuk performa maksimal 60 FPS
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0, 
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Properti Layar & Kamera
        self.W, self.H = 1280, 720
        f = self.W 
        self.cam_matrix = np.array([[f, 0, self.W/2], [0, f, self.H/2], [0, 0, 1]], dtype=np.float32)
        
        # --- INISIALISASI KUBUS 3D (BASE SHAPE KOTAK) ---
        # Kita definisikan 8 titik sudut kubus dasar [X, Y, Z]
        # Ukuran dasar rusuk 0.2m (±0.1 dari pusat)
        s = 0.1
        self.base_vertices = np.array([
            [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s], # Depan
            [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s]  # Belakang
        ], dtype=np.float32)
        
        # Definisikan rusuk untuk digambar (menghubungkan 8 titik)
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), # Sisi Depan
            (4, 5), (5, 6), (6, 7), (7, 4), # Sisi Belakang
            (0, 4), (1, 5), (2, 6), (3, 7)  # Penghubung
        ]
        
        # States Objek Spasial (Posisi, Skala, Rotasi)
        self.curr_pos = np.array([0.0, 0.0, 1.2], dtype=np.float32) # Z=1.2m (Depth)
        self.curr_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32) # Stretching [X, Y, Z]
        self.curr_angle = 0.0 # Sudut saat ini (bisa manual/auto)
        self.auto_angle = 0.0 # Sudut untuk putar otomatis
        
        # Parameter Manipulasi
        self.smoothing = 0.15 # Kehalusan LERP
        self.is_manipulating = False # State apakah tangan sedang memegang

    def lerp(self, start, end, t):
        """Interpolasi Linear untuk gerakan halus"""
        return start + t * (end - start)

    def get_rainbow_color(self, angle):
        """Menghasilkan warna BGR Pelangi yang berubah sesuai sudut"""
        hue = int(angle % 180) # Gunakan modulo 180 untuk siklus HSV penuh
        # Membuat image satu pixel HSV untuk diconvert ke BGR
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        # Konversi array numpy int64 ke standard python int tuple
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def draw_3d_cube(self, img, points_2d, color):
        """Menggambar rusuk kubus berdasarkan proyeksi titik 2D"""
        p = points_2d.astype(int)
        
        # Gambar sisi transparan (overlay) agar terlihat hologram
        overlay = img.copy()
        cv2.fillPoly(overlay, [p[:4]], color) # Sisi Depan
        cv2.fillPoly(overlay, [p[4:]], color) # Sisi Belakang
        # Gabungkan overlay transparan
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Gambar rusuk wireframe putih terang agar tajam
        for edge in self.edges:
            p1 = tuple(p[edge[0]])
            p2 = tuple(p[edge[1]])
            cv2.line(img, p1, p2, (255, 255, 255), 3)

    def run(self):
        # Setup Kamera 60 FPS
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        cap.set(cv2.CAP_PROP_FPS, 60)

        p_time = 0

        while cap.isOpened():
            success, img = cap.read()
            if not success: break
            
            # Mirroring & Konversi warna
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Proses deteksi tangan
            results = self.hands.process(img_rgb)
            
            hand_info = []
            if results.multi_hand_landmarks:
                for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    # Data tangan: Label (Left/Right) & Koordinat Ujung Jari
                    label = results.multi_handedness[hand_idx].classification[0].label
                    tips = [hand_lms.landmark[i] for i in [4, 8, 12, 16, 20]]
                    coords_2d = [(int(t.x * self.W), int(t.y * self.H)) for t in tips]
                    
                    # Hitung pusat tangan & rentangan jari (untuk mendeteksi tarikan)
                    center_2d = np.mean(coords_2d, axis=0).astype(int)
                    span_2d = np.hypot(coords_2d[0][0] - coords_2d[4][0], coords_2d[0][1] - coords_2d[4][1])
                    
                    hand_info.append({'label': label, 'center': center_2d, 'span': span_2d, 'coords': coords_2d})

            # --- LOGIKA INTERAKSI SPASIAL (MANUAL/AUTO) ---
            target_pos = self.curr_pos.copy()
            target_scale = np.array([1.0, 1.0, 1.0]) # Default scale (kotak sempurna)

            if len(hand_info) == 2:
                # 2 TANGAN TERDETEKSI: MASUK MODE MANUAL CONTROL
                self.is_manipulating = True
                h_l = next((h for h in hand_info if h['label'] == 'Left'), hand_info[0])
                h_r = next((h for h in hand_info if h['label'] == 'Right'), hand_info[1])
                c_l, c_r = h_l['center'], h_r['center']
                mid_2d = (c_l + c_r) // 2
                
                # 1. POSISI & DEPTH (KELUAR-MASUK)
                # Map posisi Y tangan ke kedalaman Z (Maju-Mundur)
                normalized_y = mid_2d[1] / self.H
                target_pos[2] = 0.5 + (1.5 * normalized_y) # Depth range 0.5m - 2.0m
                
                # Update posisi X, Y mengikuti tangan
                target_pos[0] = (mid_2d[0] - self.W/2) / self.W * self.curr_pos[2]
                target_pos[1] = (mid_2d[1] - self.H/2) / self.H * self.curr_pos[2]

                # 2. STRETCHING (PENGENCANGAN KOTAK JADI BALOK)
                # Tangan kanan kontrol Skala X, tangan kiri kontrol Skala Y
                s_x = max(0.2, h_r['span'] / 180.0)
                s_y = max(0.2, h_l['span'] / 180.0)
                # Z-scale otomatis agar volume terlihat konsisten (tidak meledak)
                s_z = 1.0 / (s_x * s_y + 0.1) # Tambah small value agar tidak div by zero
                target_scale = np.array([s_x, s_y, s_z])

                # 3. MANUAL ROTATION 360 DERAJAT
                # Hitung sudut orientasi tangan kiri ke tangan kanan
                vec = c_r - c_l
                self.curr_angle = np.degrees(np.arctan2(vec[1], vec[0])) # Z-axis (Roll)
                
                # Visualisasi kursor jari neon (Cyan) saat memegang
                for h in hand_info:
                    for pt in h['coords']:
                        cv2.circle(img, pt, 5, (255, 255, 255), -1)
                        cv2.circle(img, pt, 8, (0, 255, 255), 2)
            else:
                # TANGAN DILEPAS: MASUK MODE AUTO-ROTATE
                self.is_manipulating = False
                # Kembali ke bentuk kotak sempurna
                target_scale = np.array([1.0, 1.0, 1.0])

            # --- LOGIKA AUTO-ROTATE (360 Derajat Pelangi) ---
            if not self.is_manipulating:
                self.auto_angle += 2.0 # Kecepatan putar otomatis (2 drajat per frame)
                self.curr_angle = self.auto_angle
            
            # --- SMOOTHING (LERP) UNTUK SEMUA STATE ---
            self.curr_pos = self.lerp(self.curr_pos, target_pos, self.smoothing)
            self.curr_scale = self.lerp(self.curr_scale, target_scale, self.smoothing)

            # --- TRANSFORMASI MESH 3D ---
            # 1. Terapkan Scaling/Stretching ke 8 titik kubus dasar
            v_deformed = self.base_vertices * self.curr_scale
            
            # 2. Terapkan Rotasi 360 derajat (Manual atau Auto)
            r = R.from_euler('z', self.curr_angle, degrees=True).as_matrix()
            v_rotated = np.dot(v_deformed, r.T)
            
            # 3. Terapkan Posisi (Keluar-Masuk & Geser)
            v_final = v_rotated + self.curr_pos
            
            # 4. Proyeksikan titik-titik 3D yang sudah dideformasi ke layar 2D
            pts2d, _ = cv2.projectPoints(v_final, np.zeros((3,1)), np.zeros((3,1)), 
                                         self.cam_matrix, np.zeros((4,1)))
            
            # --- RENDER KOTAK BERWARNA (PELANGI DINAMIS) ---
            # Dapatkan warna pelangi sesuai sudut rotasi saat ini
            dynamic_color = self.get_rainbow_color(self.curr_angle)
            self.draw_3d_cube(img, pts2d.squeeze(), dynamic_color)

            # --- UI & FPS ---
            curr_time = time.time()
            fps = 1 / (curr_time - p_time) if (curr_time - p_time) > 0 else 0
            p_time = curr_time
            cv2.putText(img, f"60 FPS | Cube Mode | Angle: {int(self.curr_angle % 360)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Panduan Gestur
            guide_y = self.H - 60
            cv2.putText(img, "2 TANGAN (Maju-Mundur): Gerakkan KELUAR/MASUK", (20, guide_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, "TANGAN KANAN (X-axis) / KIRI (Y-axis): Tarik/Regangkan Kotak Jadi Balok (Stretching)", 
                        (20, guide_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, "PUTAR TANGAN: Rotasi Full 360 Derajat Manual | LEPAS TANGAN: Auto-Rotate Pelangi", 
                        (20, guide_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Spatial Master Cube v4", img)
            if cv2.waitKey(1) & 0xFF == 27: # ESC untuk keluar
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SpatialAutoCube().run()