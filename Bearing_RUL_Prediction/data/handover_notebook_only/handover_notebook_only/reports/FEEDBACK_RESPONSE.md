# Báo cáo sửa theo Feedback.md

Lượt nghiệm thu mới: 38 tests PASS trong mỗi notebook; chạy hai kernel mới ở thư mục sạch, không có file .py. Cache được xây lại từ raw trên output mới: `C:\Users\ACER\Downloads\Bearing_RUL_Prediction\artifacts\pronostia_feedback_clean_20261006_174357`. Validation 7 bộ split PASS, không có lỗi ERROR; `rebuilt=true`.

Trạng thái preprocessing: **PASS**. Trạng thái nghiệm thu benchmark: **REVIEW_REQUIRED_GROUND_TRUTH**. Không dùng báo cáo cũ để thay thế lượt chạy này.

| Feedback | Đã sửa / bằng chứng |
|---|---|
| Cách tìm project trong ảnh | Bỏ dò DATA_PREPROCESSING_PLAN.md, bỏ register_cell_magic và %%module_code. Root là working directory của bộ bàn giao. Kế hoạch chỉ là tài liệu tham chiếu. |
| Một nhiệm vụ mỗi cell | Ô định nghĩa hàm riêng; Markdown ngay phía trên. Parser tách delimiter/header/numeric/schema/clock. Pipeline tách scan, kiểm terminal, labels, temperature, vibration, scaler, windows, validation, benchmark, plots và export. notebook_structure_check.json kiểm cú pháp và cấu trúc. |
| Notebook độc lập | Hai notebook đặt chung thư mục, dùng Python thông thường; định nghĩa/tests nạp từ data_process.ipynb, không import src/scripts/tests hay discover tests ngoài. notebook_execution.json chứng minh chạy từ bản copy sạch. |
| Tests tự chứa | Nội dung parser/contract/identity/feedback/integration tests ở data_process.ipynb. notebook_test_report.json ghi 38 tests, không thất bại. |
| Config và đường dẫn | pronostia_config.yaml được gửi kèm. RAW_ROOT và OUTPUT_ROOT khai báo ở một ô; có thể đặt biến môi trường. Không chứa tên người dùng cố định trong logic. |
| Clean rebuild | Mặc định clean_rebuild; output mới/rỗng, từ chối output đã có dữ liệu, không tự xóa. validate_existing không ghi đè scaler/split/index/identity/recipe để che cache cũ. clean_rebuild_execution.json ghi rebuilt=true. |
| Terminal completeness | Pin 6 chỉ số acquisition cuối learning; terminal thiếu hoặc bị quarantine phải fail trước labels. Tests kiểm cả hai trường hợp. |
| Ground truth | CSV đủ 11 bearing, giá trị/nguồn/source_locator/verified flag. Audit Full_Test_Set ở ground_truth_full_test_comparison.csv, không dùng Full_Test_Set làm feature input. |
| Dataset/DataLoader | Tách dataset_kwargs và loader_kwargs; test inference_only, max_cached_bearings và drop_last. validation_report.json có full-pass và trace batch thật của train/validation/test. |
| Mask và causal alignment | Kiểm dtype bool, shape, masked temperature=0 sau scale; delta không âm, gap; test sửa nhiệt độ tương lai không đổi history. |
| Temperature coverage | temperature_coverage.csv có file counts, đầu/cuối timestamp, aligned counts/%, missing %. temperature_report.json có delta quantiles. modality_protocol.json có folds bị ảnh hưởng và quy tắc so sánh vibration-only/multimodal. |
| Source pin | YAML pin commit + SHA256 archive; kiểm cả ZIP commit comment. dataset_pin_verification.json xác minh ZIP thật; downloader từ chối snapshot đổi nội dung. |
| Legacy helpers | Loại khỏi code/test nghiệm thu notebook. Các .py cũ là lịch sử; converter cũ ngừng dùng để tránh ghi đè notebook. |
| Reports mới | Quality, labels, temperature, QC, validation, coverage, manifest summary, provenance và bằng chứng Run All đều lấy từ lượt chạy mới. |

## Điểm còn cần leader xác minh

10/11 ground truth khớp Full_Test_Set terminal-minus-cut. **Bearing1_4: nguồn xuất bản ghi 339 s, terminal-minus-cut là 2890 s, chênh 2551 s.** Giữ nguyên nhãn nguồn, xuất cảnh báo và đặt `official_benchmark_ground_truth_ready=false`. Bổ sung kiểm tra mục 5: tài liệu BTC Table 3/trang 8 xác nhận 339 s; Note 1/trang 6 định nghĩa failure là accelerometer vượt 20g. Vẫn chưa giải thích được mismatch với raw endpoint. Xem ROLE1_SECTION5_ACCEPTANCE.md; giữ official benchmark chưa sẵn sàng.

Bearing hoàn toàn thiếu nhiệt độ: Bearing2_2, Bearing3_2, Bearing1_3, Bearing2_3, Bearing2_6. Không tạo nhiệt độ giả; giữ mask. So sánh các mô hình phải dùng cùng folds/windows/target và công bố coverage, đặc biệt trên official test.

## Gửi leader

Gửi toàn bộ `handover_notebook_only/` hoặc ZIP cùng tên. Hai notebook, YAML, ground truth CSV, requirements, README, kế hoạch tham chiếu và reports đều ở đó. Không cần gửi .tmp/src/scripts/tests, raw hoặc cache lớn cùng ZIP.

Raw vẫn ngoài project tại `C:\Users\ACER\Downloads\PHM_2012_Raw\ieee-phm-2012-data-challenge-dataset-master`. Cache mới ở output nêu đầu báo cáo; muốn bàn giao dữ liệu tensor để train ngay thì cấp leader đường dẫn/copy riêng cache này. Báo cáo chính để đọc nhanh là **FEEDBACK_RESPONSE.md**, bằng chứng máy đọc là **clean_rebuild_execution.json** và **validation_report.json**.
