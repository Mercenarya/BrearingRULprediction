# Nghiệm thu mục 5 - ROLE1_VERIFICATION_REQUIREMENTS.md

**Kết luận: đã xác nhận nhãn 339 giây trong tài liệu BTC; chưa giải thích được chênh lệch endpoint của bản raw. Đáp ứng nhánh dự phòng mục 5, không mở nghiệm thu metric official.**

## Nguồn đủ tin cậy cho nhãn và định nghĩa failure

Tài liệu **IEEE PHM 2012 Prognostic challenge — Outline, Experiments, Scoring of results, Winners** của BTC, được lưu cùng dataset đã dùng:

- **Mục 5.2, Table 3, trang 8:** bảng Actual RULs to be estimated ghi Bearing1_4 bằng **339 s**. Cả 11 giá trị trong test_ground_truth.csv đều khớp bảng này.
- **Mục 3.1, trang 5 và Note 1, trang 6:** RUL là thời gian tới khi accelerometer vượt **20g**. Tài liệu mô tả việc dừng thí nghiệm để bảo vệ bệ thử khi biên độ rung vượt ngưỡng này.
- **Mục 4.1, trang 6:** vibration được ghi 0,1 giây mỗi 10 giây; không phải tín hiệu liên tục được lưu trọn vẹn.

[Bản tài liệu theo commit đã pin](https://raw.githubusercontent.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset/129d43e0aa71e1da992bb4c0f69d54fd990c58cf/IEEEPHM2012-Challenge-Details.pdf). Bản PDF nguyên gốc nằm tại `reports/IEEEPHM2012-Challenge-Details.pdf`; đã mở và kiểm tra trực quan trang 6 và 8. Nguồn này là tài liệu BTC được lưu trên mirror GitHub; không phải xác nhận mới từ máy chủ BTC.

SHA-256 PDF: `13bd43eb2dc4c8455e36584094552ccbd8101b9a0b99bd10ca82103453585ba5`. Hash bản PDF local khớp PDF trong ZIP dataset đã pin commit và checksum. Không cần dùng bài báo thứ cấp làm nguồn duy nhất nữa.

## Đối chiếu raw thực tế

| Đại lượng | Kết quả |
|---|---:|
| Acquisition Test_set/Bearing1_4 | 1139 |
| Acquisition Full_Test_Set/Bearing1_4 | 1428 |
| Timestamp đầu acquisition tại điểm cắt | 40660.42504 s |
| Timestamp đầu acquisition cuối Full_Test_Set | 43550.42504 s |
| Terminal-minus-cut | 2890 s |
| Target trong bảng BTC và CSV/config | 339 s |
| Chênh lệch | 2551 s |

Đã so sánh numeric array của **tất cả 1139 file test** với file cùng chỉ số trong Full_Test_Set: khớp hoàn toàn. Byte/checksum file có thể khác vì Test_set dùng dấu phẩy, Full_Test_Set dùng dấu chấm phẩy và cách biểu diễn số mũ khác. Khác định dạng CSV không giải thích chênh lệch thời gian.

Kiểm tra phụ dùng absolute peak trên hai kênh rung cho thấy acquisition **1126** đã vượt 20g, tức **130 giây trước điểm cắt**; kênh 1 có peak **23,427g**. Ở acquisition 1139, peak hai kênh là **20,953g** và **24,243g**. Vì vậy, lấy “file đầu tiên có abs peak >20g” không tái tạo target 339 s trên bản raw này.

Đây là kiểm tra chẩn đoán, **không phải thuật toán failure trigger do BTC công bố**. Tài liệu không đủ chi tiết để suy ra thời điểm trigger thực tế từ các đoạn ghi thưa; cũng chưa có bằng chứng giải thích vì sao bản Full_Test_Set này còn 2890 s. Không kết luận 339 là typo, không đổi thành 2890, không khẳng định chắc chắn đã thu thêm dữ liệu sau failure.

Chi tiết từng acquisition: `bearing1_4_raw_endpoint_audit.csv`. Kết quả máy đọc: `role1_section5_verification.json`.

## Kiểm tra nhánh dự phòng bắt buộc

| Yêu cầu mục 5 | Kết quả nghiệm thu |
|---|---|
| Giữ nguyên cảnh báo khi chưa giải thích được mâu thuẫn | PASS: vẫn giữ cảnh báo 2551 s và trạng thái REVIEW_REQUIRED_GROUND_TRUTH |
| Không tự đổi nhãn | PASS: Bearing1_4 vẫn bằng 339 s; không đổi nhãn bất kỳ bearing nào |
| Giữ official_benchmark_ground_truth_ready=false | PASS: còn false trong clean_rebuild_execution.json, validation_report.json và báo cáo mục 5 |
| Không báo metric official trên test set | PASS: không huấn luyện/chấm điểm official; report validation chỉ có kiểm dữ liệu và tốc độ DataLoader |

Phân biệt **verified_against_primary_source=true** (khớp bảng BTC) với **verified_against_full_test=false** của Bearing1_4 (không khớp terminal-minus-cut). Không dùng trạng thái xác nhận nguồn để tự mở cổng official benchmark.

## Phạm vi của lượt kiểm tra này

Đây là bổ sung bằng chứng nguồn và kiểm endpoint cho **mục 5**. Không sửa code preprocessing, cấu hình xử lý, split, nhãn số hay tensor cache; không chạy một lượt clean rebuild mới. Bằng chứng 38 tests PASS và 6 fold + official DataLoader PASS vẫn là lượt clean rebuild đã ghi trong clean_rebuild_execution.json. CSV được bổ sung các cột nguồn BTC, giữ nguyên các giá trị và cờ đối chiếu Full_Test_Set.

**Mục 5 đạt nhánh dự phòng. Đối chiếu failure endpoint chưa khép lại; official benchmark vẫn chưa sẵn sàng.** Báo cáo này không xác nhận đã thực thi trên Google Colab, không thay thế các điều kiện khác trong file yêu cầu.
