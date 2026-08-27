USE db_skindetect;

ALTER TABLE prediksi
ADD COLUMN image_url VARCHAR(1000) NULL;

CREATE TABLE IF NOT EXISTS kelas_penyakit (
    id_kelas INT AUTO_INCREMENT PRIMARY KEY,
    nama_kelas VARCHAR(100) NOT NULL,
    deskripsi TEXT
);

INSERT INTO kelas_penyakit
(id_kelas, nama_kelas, deskripsi)
VALUES
(1, 'Chickenpox', 'Hasil klasifikasi mengarah pada Chickenpox atau cacar air.'),
(2, 'Dermatitis Eksim', 'Hasil klasifikasi mengarah pada Dermatitis atau Eksim.'),
(3, 'Measles', 'Hasil klasifikasi mengarah pada Measles atau campak.'),
(4, 'Monkeypox', 'Hasil klasifikasi mengarah pada Monkeypox.')
ON DUPLICATE KEY UPDATE
nama_kelas = VALUES(nama_kelas),
deskripsi = VALUES(deskripsi);

CREATE TABLE IF NOT EXISTS prediksi (
    id_prediksi INT AUTO_INCREMENT PRIMARY KEY,
    id_kelas INT NOT NULL,
    nama_gambar VARCHAR(255) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    tanggal_prediksi DATETIME DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_prediksi_kelas
    FOREIGN KEY (id_kelas)
    REFERENCES kelas_penyakit(id_kelas)
    ON UPDATE CASCADE
    ON DELETE RESTRICT
);
