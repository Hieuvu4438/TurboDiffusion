import pandas as pd
import numpy as np

def create_lfw_balanced_csv(input_csv, output_csv):
    # 1. Đọc dữ liệu gốc (Match pairs)
    df_match = pd.read_csv(input_csv)
    df_match['label'] = 1
    df_match['person_ref'] = df_match['person'] # Cùng người
    
    # 2. Tạo tập Mismatch (Negative pairs)
    df_mismatch = df_match.copy()
    df_mismatch['label'] = 0
    
    # Dịch chuyển cột ref_path và person để tạo cặp sai
    # Người A sẽ được so sánh với ảnh ref của người B
    df_mismatch['ref_path'] = np.roll(df_mismatch['ref_path'], 1)
    df_mismatch['person_ref'] = np.roll(df_mismatch['person'], 1)
    
    # 3. Gộp lại và xáo trộn
    df_final = pd.concat([df_match, df_mismatch], ignore_index=True)
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    
    # 4. Lưu file
    df_final.to_csv(output_csv, index=False)
    print(f"Đã tạo dataset cân bằng: {output_csv}")
    print(f"Thống kê: {len(df_match)} Match + {len(df_mismatch)} Mismatch = {len(df_final)} mẫu.")

# Chạy lệnh tạo file
create_lfw_balanced_csv("csv/side/downup_L3.csv", "csv/side/negativeSample/downup_L3_negative.csv")