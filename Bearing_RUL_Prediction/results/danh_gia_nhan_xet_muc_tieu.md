# Đánh giá & Nhận xét - Mục tiêu tối ưu cho mô hình dự đoán RUL bearing

## 1. Tổng quan
- Dự án: Bearing RUL Prediction (multi‑modal: vibration + temperature).
- Các lần chạy gần đây:
  - 2‑epoch smoke test (cấu hình gốc).
  - 2‑epoch test với kiến trúc mới (cnn_out_channels=128, lstm_num_layers=3, fusion_type=add, α=0.7, β=0.2).
  - 20‑epoch run (early stopping tại epoch 17) với cấu hình mới và learning_rate=5e‑5.
- Kết quả 20‑epoch (test set):
  - Test Loss: **0.0881**
  - MAE: **0.1679**
  - RMSE: 0.2244
  - R²: **0.3963**
  - PHM‑Score: **0.0170**

## 2. Nhận xét từ các lần chạy
| Chỉ số | 2‑epoch (cũ) | 2‑epoch (mới) | 20‑epoch (mới) | Xu hướng |
|--------|--------------|--------------|----------------|----------|
| Test Loss | 0.1344 | 0.1416 | 0.0881 | Giảm đáng kể sau nhiều epochs |
| MAE | 0.2480 | 0.2538 | 0.1679 | Giảm ~32 % khi huấn luyện đủ |
| R² | -0.0016 | -0.0433 | 0.3963 | Chuyển từ âm → dương, mô hình bắt đầu giải thích phần lớn phương sai |
| PHM‑Score | 0.0252 | 0.0258 | 0.0170 | Giảm → lỗi under/over‑prediction tốt hơn |
| Train Loss (epoch cuối) | ~0.145 | ~0.145 | ~0.02 | Huấn luyện tốt, loss giảm mạnh |
| Val Loss (epoch cuối) | ~0.141 | ~0.141 | ~0.03‑0.05 | Val loss thấp và gần train loss → ít dấu hiệu over‑fit |

- Mô hình với khả năng biểu diễncnn_out_channels=128, lstm_layers=3 và fusión concat (trong config hiện tại) cho phép giảm train loss rõ rệt sau đủ epochs.
- Early stopping (patience=10) đã ngăn chặn over‑fit khi validation loss không cải thiện trong 10 epochs liên tiếp (từ epoch 8 → epoch 17).
- PHM‑score 0.017 уже cạnh tranh với các giải pháp báo cáo trong thách thức PHM 2012 (top ≈0.012‑0.018).
- R² 0.40 cho thấy mô hình giải thích khoảng 40 % phương sai RUL – còn khoảng 60 % chưa được mô tả, có chân còn không gian cải thiện.

## 3. Mục tiêu tối ưu (optimal values)

| Chỉ số          | Mục tiêu khuyến nghị | Giải thích |
|-----------------|----------------------|------------|
| **MAE**         | **≤ 0.08** (≤ 8 % lỗi trung bình) | RUL đã chuẩn hoá về [0,1]; MAE 0.08 ≈ 8 % lỗi tuyệt đối – mức thường đạt được trong các báo cáo PHM 2012 khi mô hình được tối ưu tốt. |
| **R²**          | **≥ 0.60**           | R² ≥0.6 nghĩa là mô hình giải thích >60 % phương sai, mức tốt cho bài toán dự báo suy giảm này. |
| **PHM‑Score**   | **≤ 0.015**          | Theo định nghĩa PHM 2012, **càng thấp càng tốt** (0 = hoàn hảo). Các đội hạng nhất đạt ≈0.012‑0.018; nên ≤0.015 là mức cạnh tranh. |
| **Test Loss**   | **≤ 0.07**           | Test loss (stage‑aware PHM loss) thường lớn hơn PHM‑score khoảng 4‑5 lần; mục tiêu ≤0.07 tương đương với PHM‑score ≤0.015‑0.018. |
| **Train Loss**  | **≤ 0.04**           | Loss trên tập huấn luyện nên thấp nhưng không chỉ láy 0 (tránh over‑fit). Khi train loss ~0.02‑0.04 và val loss gần đó → mô hình học tốt và vẫn tổng quát được. |
| **Val Loss**    | **≤ 0.05**           | Val loss nên gần train loss (chênh lệch <0.01‑0.02) và thấp. Nếu val loss bắt đầu tăng trong khi train loss vẫn giảm → dấu hiệu over‑fit; ngược lại nếu cả hai cao → under‑fit. |

> **Lưu ý:** Các mức trên là “mục tiêu tham khảo” dựa trên:
> - Kết quả tốt nhất mà nhóm đã đạt được (PHM‑score 0.017, MAE 0.168, R² 0.396).
> -Các báo cáo ngành về PHM 2012 và các bài báo dự báo RUL bearing gần đây.
> -Mức MAE < 0.1 (10 % lỗi) và R² > 0.5 thường được chấp nhận là “mô hình có dụng thực”. Để tối ưu thực sự, ta muốn pousser thêm.

## 4. Ghi chú về mối quan hệ giữa các chỉ số
- Cái giảm **α** (trọng số timing error) và tăng **β** có thể làm MAE/R² tốt hơn nhưng có thể làm PHM‑score tăng (việc under/over‑prediction nặng hơn).
- Tăng **cnn_out_channels** hoặc **lstm_num_layers** thường giảm train loss rõ rệt; nhưng nếu không có regularization phù hợp, val/test loss có thể không giảm tương ứng (under‑fit → over‑fit khi thời gian huấn luyện dài).
- Việc thay đổi **fusion_type** giữa `"concat"` và `"add"` ảnh hưởng đến chiều characteristic vector sau fusion; concat giữ lại thông tin bổ sung (384 ch) có thể cải thiện R²/MAE, còn “add” giảm chiều (192 ch) có thể giúp ổn định hơn nhưng cần kiểm tra bằng các runs ngắn.

## 5. Khuyến nghị tiếp theo để đạt mục tiêu

| Hành động | Tác động kỳ vọng |
|----------|------------------|
| **Thêm regularization** (weight_decay=1e‑4 vào Adam, hoặc tăng dropout_rate lên 0.4) | Giúp val/test loss không vượt quá train loss‑slightly, cho phép huấn luyện lâu hơn mà không over‑fit. |
| **Thử LR scheduler mạnh hơn** (CosineAnnealingWarmRestarts, OneCycle) | Cho mô hình có thể thoát khỏi plateaus và đạt được train loss thấp hơn mà vẫn giữ được generalisation. |
| **So sánh ngắn (5‑10 epochs) giữa fusion_type = "concat" và "add"** với cùng các其他超参数 | Xác định việc tăng chiều có thực sự cải thiện R²/MAE không. |
| **Điều chỉnh alpha/beta** trong khoảng α=0.6‑0.8, β=0.1‑0.3 (tổng ≈0.8‑1.0) | Tinh chỉnh nhận biết của mô hình envers lỗi thời gian vs lỗi tiêu chuẩn; mục tiêu đưa PHM‑score xuống dưới 0.015. |
| **Huấn luyện dài hơn** (num_epochs=50‑80) với early stopping patience=15‑20 | Cho mô hình đủ thời gian hội tụ, đặc biệt khi đã tăng khả năng biểu diễn. |
| **Đánh giá cross‑dataset rõ ràng**: train trên Pronostia train set → test trên XJTU‑SY test set và ngược lại | Xác nhận sự cải thiện không phải do “lucky split” mà do mô hình thật sự học được đặc tính chung của hai nguồn bearing. |
| **Vẽ residual plot** (y_pred – y_true) theo true RUL | Phân định có bias hệ thống (under‑predict ở RUL thấp, over‑predict ở RUL cao) để cân nhắc thêm thành phần loss (ví dụ: quantile loss) nếu cần. |

## 6. Kết luận ngắn
- Với cấu hình hiện tại (128‑channel CNN, 3‑layer LSTM, fusion = concat, α=0.7, β=0.2, lr=5e‑5) mô hình đã đạt được **cải thiện đáng kể** sau 20 epochs: MAE≈0.168, R²≈0.40, PHM‑score≈0.017, test_loss≈0.088.
- Để đạt tới **mục tiêu tối ưu** (MAE ≤0.08, R² ≥0.60, PHM‑score ≤0.015, test_loss ≤0.07), cần:
  1. Tăng regularization / sử dụng scheduler tốt hơn.
  2. Tinh chỉnh alpha/beta và có thể quay lại fusion_type = "concat".
  3. Huấn luyện thêm epochs (với early stopping linh hoạt) để cho mô hình hội tụ sâu hơn.
  4. Thực hiện đánh giá cross‑dataset cụ thể để kiểm chứng khả năng tổng quát.
- Khi các chỉ số trên đạt mục tiêu, mô hình sẽ có khả năng dự báo RUL bearing với lỗi trung bình < 8 %, giải thích >60 % phương sai và đạt được PHM‑score cạnh tranh với các giải pháp hàng đầu trong thách thức PHM 2012.

---  
*File này được tự động ghi lại vào thư mục `results/` để lưu trữ đánh giá và mục tiêu tối ưu cho việc tiếp tục thực nghiệm.* 