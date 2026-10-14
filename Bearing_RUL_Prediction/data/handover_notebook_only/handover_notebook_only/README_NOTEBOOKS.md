# Bàn giao notebook-only sau Feedback.md

Hai notebook cùng thư mục là source of truth. Không cần src/, scripts/, tests/ hoặc bất kỳ file .py nào để chạy bộ này. DATA_PREPROCESSING_PLAN.md là tài liệu tham chiếu, không dùng để xác định project root.

## Cài dependencies

Python 3.11; cài `pip install -r requirements.txt`. Chọn kernel Python 3.11 trong Jupyter/VS Code.

## Run All

1. Đặt RAW_ROOT (root có Learning_set, Test_set, Full_Test_Set) và OUTPUT_ROOT mới trong ô cấu hình đường dẫn hoặc bằng biến môi trường PRONOSTIA_RAW_ROOT và PRONOSTIA_OUTPUT_ROOT.
2. Mở data_pipeline_scripts.ipynb từ thư mục có hai notebook và YAML/CSV này. Restart Kernel rồi Run All.
3. Mặc định clean_rebuild. Output không được chứa dữ liệu; notebook từ chối nếu thư mục không rỗng và không tự xóa thư mục.
4. Code preprocessing và test implementation được nạp từ data_process.ipynb cùng thư mục. Không cần chạy notebook data bằng tay trước.
5. Check reports/clean_rebuild_execution.json: rebuilt=true, preprocessing_status=PASS. Các cảnh báo ground truth/modality phải được đọc riêng.

Mode validate_existing chỉ dùng để kiểm cache đã xây: đặt PRONOSTIA_EXECUTION_MODE=validate_existing và PRONOSTIA_OUTPUT_ROOT tới cache đó. Đây không phải bằng chứng clean rebuild mới.

## Nguồn và ground truth

Dataset mirror: https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset . Commit và expected SHA-256 trong YAML. URL archive master được pin theo cả SHA-256 và ZIP commit comment; downloader từ chối bất kỳ byte snapshot mới nào. Archive đang có từ lần tải đầu đã được kiểm hash/commit; notebook mặc định dùng raw sẵn có, không tải lại.

Ground truth ban đầu dẫn Table 2 của bài báo DOI 10.3390/app8071102. Đã bổ sung đối chiếu nguồn BTC: IEEEPHM2012-Challenge-Details.pdf, mục 5.2/Table 3/trang 8 xác nhận cả 11 nhãn, gồm Bearing1_4=339 s; Note 1/trang 6 định nghĩa RUL tới accelerometer vượt 20g. CSV có thêm primary_source, primary_source_locator và verified_against_primary_source. Full_Test_Set vẫn cho terminal-minus-cut=2890 s ở Bearing1_4, chưa giải thích được chênh 2551 s; giữ nhãn và official_benchmark_ground_truth_ready=false, không báo metric official. Đọc reports/ROLE1_SECTION5_ACCEPTANCE.md cùng JSON/CSV audit và PDF nguồn.

## Temperature

Giữ tất cả bearing, không tạo nhiệt độ giả; downstream phải dùng missing mask. Báo cáo coverage liệt kê missing đầy đủ/partial theo bearing và fold. Vibration-only và multimodal dùng cùng folds/windows/target; thêm đánh giá subset đủ modality phải công bố danh sách bearing.

## Bộ gửi leader

Hai notebook; pronostia_config.yaml; test_ground_truth.csv; README_NOTEBOOKS.md; requirements.txt; DATA_PREPROCESSING_PLAN.md; reports/. Không cần gửi raw hay toàn bộ cache. reports chứa bằng chứng clean rebuild, chất lượng, nhãn, coverage, QC, validation, manifest summary và provenance. Output của notebook được lưu từ lần chạy mới; thông tin máy/path trong báo cáo chỉ là provenance của lượt đó.

Báo cáo đối chiếu từng mục Feedback.md: reports/FEEDBACK_RESPONSE.md. Thông tin raw/cache thực tế của lượt nghiệm thu ở reports/clean_rebuild_execution.json.

## Google Colab

Hai notebook có cell cài dependencies, upload ZIP bàn giao, mount Drive, cấu hình /content/Drive và tải/kiểm raw. Xem COLAB_GUIDE.md. Cell cuối pipeline xuất colab_execution.json và tải reports ZIP từ lượt chạy thật.
