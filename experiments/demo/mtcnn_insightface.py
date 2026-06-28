import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from insightface.model_zoo import get_model
from numpy.linalg import norm
from skimage import transform as trans

def align_face(img, landmarks):
    """Căn chỉnh khuôn mặt về chuẩn 112x112 dựa trên 5 điểm mốc"""
    src = np.array([
        [30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
        [33.5493, 92.3655], [62.7299, 92.2041] ], dtype=np.float32)
    
    # 5 điểm mốc từ MTCNN: mắt trái, mắt phải, mũi, miệng trái, miệng phải
    dst = np.array([
        landmarks['left_eye'], landmarks['right_eye'], landmarks['nose'],
        landmarks['mouth_left'], landmarks['mouth_right']
    ], dtype=np.float32)
    
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return aligned

def get_aligned_embedding(img_path, detector, model):
    img = cv2.imread(img_path)
    if img is None: return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = detector.detect_faces(img_rgb)
    if not results: return None, None
    
    # Lấy mặt lớn nhất và thực hiện Alignment
    res = max(results, key=lambda x: x['box'][2] * x['box'][3])
    face_aligned = align_face(img_rgb, res['keypoints'])
    
    # Trích xuất đặc trưng (512-d)
    embedding = model.get_feat(face_aligned).flatten()
    return embedding, face_aligned

def run_comparison(path1, path2):
    print("Khởi tạo model...")
    detector = MTCNN()
    # model buffalo_l cực kỳ mạnh cho nhận diện
    model = get_model('buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    model.prepare(ctx_id=0) # Dùng GPU RTX 5880

    print(f"Đang xử lý và Alignment...")
    emb1, face1 = get_aligned_embedding(path1, detector, model)
    emb2, face2 = get_aligned_embedding(path2, detector, model)

    if emb1 is not None and emb2 is None:
        print("Lỗi: Không tìm thấy mặt.")
        return

    # Tính Cosine Similarity
    score = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    
    # Hiển thị kết quả (Threshold 0.75 như paper)
    is_same = score >= 0.75
    title_color = 'green' if is_same else 'red'
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(face1); plt.title("Aligned Face 1"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(face2); plt.title("Aligned Face 2"); plt.axis('off')
    plt.suptitle(f"Similarity Score: {score:.4f}\nResult: {'SAME' if is_same else 'DIFF'}", 
                 color=title_color, fontsize=16, fontweight='bold')
    plt.show()

# Thay đường dẫn của bạn
run_comparison('/home/haipd/TurboDiffusion/Experiment_Data/Side/Abdullah_Gul_side.jpg', '/home/haipd/TurboDiffusion/Experiment_Data/Ref/Abdullah_Gul_ref.jpg')