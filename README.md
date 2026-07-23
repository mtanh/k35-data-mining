# k35-data-mining

Dự án này gồm hai script Python dùng để chạy phân tích bootstrap với LiNGAM:

- `run_direct_lingam_bootstrap.py`
- `run_ica_lingam_bootstrap.py`

## Cách chạy

Mở terminal tại thư mục gốc của dự án, sau đó chạy các lệnh bên dưới.

### 1. Tạo môi trường ảo

```powershell
python -m venv .venv
```

### 2. Kích hoạt môi trường ảo

Trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 3. Cài đặt thư viện phụ thuộc

```powershell
pip install -r requirements.txt
```

### 4. Chạy script DirectLiNGAM bootstrap

```powershell
python run_direct_lingam_bootstrap.py
```

### 5. Chạy script ICA-LiNGAM bootstrap

```powershell
python run_ica_lingam_bootstrap.py
```

## Kết quả đầu ra

Mỗi script sẽ tạo thư mục kết quả riêng trong thư mục gốc của dự án:

- `outputs_direct_lingam`
- `outputs_ica_lingam`
