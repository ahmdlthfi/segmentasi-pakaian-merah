import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Baca gambar ---
img = cv2.imread('pakaian_merah.png')

# Pastikan gambar berhasil dibaca
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan! Cek nama atau path file kamu.")

# --- 2. Gaussian Blur (haluskan noise besar) ---
gaussian = cv2.GaussianBlur(img, (7, 7), 0)

# --- 3. Median Filter (hilangkan noise kecil tanpa kaburkan tepi) ---
median = cv2.medianBlur(gaussian, 5)

# --- 4. Bilateral Filter (halus tapi tepi tetap tajam) ---
bilateral = cv2.bilateralFilter(median, d=9, sigmaColor=75, sigmaSpace=75)

# --- 5. Konversi ke HSV ---
hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)

# --- 6. Rentang warna merah ---
lower_red1 = np.array([0, 120, 80])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 80])
upper_red2 = np.array([180, 255, 255])

# Gabungkan dua range merah (karena merah melintasi hue 0/180)
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# --- 7. Bersihkan mask dengan Morphological Filter ---
kernel = np.ones((5,5), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# --- 8. Terapkan mask ke gambar asli ---
result = cv2.bitwise_and(img, img, mask=mask_clean)

# --- 9. Tampilkan hasil ---
plt.figure(figsize=(16,6))
plt.subplot(1,5,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Gambar Asli')
plt.subplot(1,5,2); plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)); plt.title('Gaussian Filter')
plt.subplot(1,5,3); plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB)); plt.title('Median Filter')
plt.subplot(1,5,4); plt.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)); plt.title('Bilateral Filter')
plt.subplot(1,5,5); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title('Hasil Segmentasi')
plt.tight_layout()
plt.show()
