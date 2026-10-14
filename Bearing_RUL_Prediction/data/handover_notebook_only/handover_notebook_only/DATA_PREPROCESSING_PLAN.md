# Kế Hoạch Triển Khai Phần 1: Data Ingestion, Đồng Bộ Và Tiền Xử Lý PRONOSTIA

**Dự án:** Dự đoán Remaining Useful Life (RUL) của vòng bi bằng dung hợp rung động và nhiệt độ
**Dataset chính:** PRONOSTIA / IEEE PHM 2012 Prognostic Challenge
**Phạm vi tài liệu:** Phần 1 - xây dựng dữ liệu đầu vào có thể tái lập cho các baseline và mô hình deep learning
**Trạng thái:** Đặc tả giao việc và nghiệm thu; chưa phải báo cáo kết quả thực nghiệm

---

## 1. Mục Tiêu Và Nguyên Tắc Bắt Buộc

Đầu ra của Phần 1 phải là một pipeline chạy lại được từ CSV thô, tạo ra manifest, split, scaler, nhãn RUL, các window đồng bộ và cache dùng được trực tiếp bởi PyTorch.

Các nguyên tắc không được vi phạm:

1. **Không chia ngẫu nhiên theo window.** Mọi split phải theo `bearing_id`.
2. **Không dùng `Test_set` để chọn siêu tham số, early stopping, fit scaler hoặc chọn cách tiền xử lý.**
3. **Không để tùy chọn debug như `max_files` làm thay đổi tuổi thọ thật hoặc nhãn RUL.**
4. **Không nội suy một giá trị nhiệt độ thành 64 bản sao rồi coi là tín hiệu mới.**
5. **Không fit scaler riêng trên từng bearing validation/test hoặc trên toàn bộ dataset.**
6. **Không giả định schema, delimiter hoặc header trước khi kiểm tra file thực tế.**
7. **Mọi phép biến đổi phải có cấu hình, seed, version và test tự động.**

---

## 2. Dữ Liệu Chuẩn Và Phạm Vi Sử Dụng

### 2.1 Cấu trúc benchmark

- `Learning_set`: 6 bearing chạy đến điểm dừng hỏng, gồm `Bearing1_1`, `Bearing1_2`, `Bearing2_1`, `Bearing2_2`, `Bearing3_1`, `Bearing3_2`.
- `Test_set`: 11 bearing bị cắt trước điểm dừng hỏng, gồm `Bearing1_3` đến `Bearing1_7`, `Bearing2_3` đến `Bearing2_7`, và `Bearing3_3`.
- `Full_Test_Set` nếu có: chỉ dùng để kiểm chứng/khôi phục ground truth theo protocol đã phê duyệt; tuyệt đối không đưa vào train hoặc validation của benchmark chính.
- Vibration: hai trục gia tốc ngang và đứng; một snapshot thường có 2.560 mẫu trong 0,1 giây ở 25,6 kHz, được ghi định kỳ khoảng 10 giây.
- Temperature: phải đọc và xác nhận schema/tần suất từ file thực tế; không hard-code delimiter hoặc số dòng dựa trên notebook cũ.

### 2.2 Ba điều kiện vận hành

| Condition | Tốc độ |    Tải | Bearings                            |
| --------- | --------: | ------: | ----------------------------------- |
| OC1       | 1.800 rpm | 4.000 N | `Bearing1_1` đến `Bearing1_7` |
| OC2       | 1.650 rpm | 4.200 N | `Bearing2_1` đến `Bearing2_7` |
| OC3       | 1.500 rpm | 5.000 N | `Bearing3_1` đến `Bearing3_3` |

Điều kiện vận hành phải là metadata đầu vào của mọi sample để hỗ trợ đánh giá single-condition, mixed-condition và cross-condition.

---

## 3. Luồng Xử Lý Chuẩn

```text
Raw dataset
  -> inventory + checksum + schema discovery
  -> canonical manifest theo bearing và timestamp
  -> định nghĩa protocol split theo bearing
  -> parse/clean vibration và temperature
  -> đồng bộ hai modality theo timestamp
  -> tạo nhãn RUL tuyệt đối
  -> fit preprocessing/scaler chỉ trên train bearings
  -> transform từng acquisition
  -> tạo sequence window bên trong từng bearing
  -> export cache + PyTorch Dataset/DataLoader
  -> chạy validation report và unit/integration tests
```

Split được xác định trước khi fit filter parameter dựa trên dữ liệu, scaler, clipping threshold, FPT detector hoặc bất kỳ thống kê học từ dữ liệu nào.

---

## 4. Data Contract Chuẩn

Pipeline phải tạo `manifest.parquet` hoặc `manifest.csv` với tối thiểu các cột:

| Trường                        | Ý nghĩa                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `bearing_id`                  | Ví dụ`Bearing1_1`                                                           |
| `condition_id`                | `OC1`, `OC2`, `OC3`                                                       |
| `subset`                      | `Learning_set`, `Test_set`, `Full_Test_Set`                               |
| `acquisition_index`           | Chỉ số snapshot rung sau khi sort theo số trong filename                     |
| `acc_path`, `temp_path`     | Đường dẫn tương đối từ data root; temp có thể null trước alignment |
| `acc_timestamp`               | Timestamp parse từ nội dung file hoặc timestamp chuẩn hóa                  |
| `temp_timestamp`              | Timestamp temperature được ghép vào acquisition                            |
| `alignment_delta_s`           | Sai lệch thời gian giữa hai modality                                         |
| `rpm`, `load_n`             | Điều kiện vận hành                                                         |
| `n_acc_samples`               | Số mẫu rung thực tế                                                         |
| `data_status`                 | `ok`, `warning`, `quarantined`                                            |
| `rul_seconds`, `rul_cycles` | Nhãn tuyệt đối; null nếu chưa được phép tạo                          |
| `split`, `fold`             | Train/validation/test và fold tương ứng                                     |

Tất cả filename phải được **natural sort** theo chỉ số số học; không dùng lexicographic sort thuần túy.

---

## 5. Các Task Thực Hiện Và Điều Kiện Nghiệm Thu

### Task 1: Inventory, Schema Discovery Và Integrity Check

**Công việc**

- Quét `Learning_set`, `Test_set` và `Full_Test_Set` nếu tồn tại.
- Tạo checksum hoặc ít nhất `size + modified time` cho mỗi file để cache có thể phát hiện dữ liệu nguồn thay đổi.
- Tự động phát hiện delimiter, header và số cột của từng loại file.
- Với vibration, xác định đúng các cột timestamp, horizontal acceleration và vertical acceleration. Không dùng schema 5 cột nếu file thực tế có thêm microsecond/millisecond.
- Kiểm tra file rỗng, dòng lỗi, NaN/Inf, số mẫu, timestamp không tăng, index thiếu/trùng và file bị trùng nội dung.
- File lỗi phải được ghi vào `data_quality_report.json`; không được silently skip.

**Nghiệm thu**

- [ ] Manifest liệt kê 100% file tìm thấy và tổng số file theo từng bearing/modality.
- [ ] Mỗi file có schema, số dòng và trạng thái kiểm tra.
- [ ] Vibration hợp lệ có hai kênh và số mẫu kỳ vọng; mọi ngoại lệ được quarantine kèm lý do.
- [ ] Parser có test với ít nhất: delimiter khác nhau, có/không header, file rỗng, thiếu cột, NaN và filename không liên tục.
- [ ] Báo cáo tổng hợp không phụ thuộc vào thứ tự duyệt thư mục của hệ điều hành.

### Task 2: Split Protocol Theo Bearing

Triển khai hai protocol tách biệt:

**A. Development protocol**

- Chỉ dùng 6 bearing trong `Learning_set`.
- Dùng Group/Leave-One-Bearing-Out cross-validation để lựa chọn preprocessing, model và hyperparameter.
- Báo cáo fold phải nêu rõ bearing train và validation.
- Không được dùng `Bearing1_3`, `Bearing2_3` hoặc bất kỳ bearing nào của `Test_set` làm validation.

**B. Official benchmark protocol**

- Sau khi khóa toàn bộ quyết định bằng development protocol, fit lại trên cả 6 learning bearings.
- Đánh giá đúng 11 truncated test bearings bằng ground-truth RUL chính thức tại điểm cắt.
- Nếu dùng `Full_Test_Set` để suy ra ground truth, phải tách code/path rõ ràng và có cờ cấu hình; dữ liệu sau điểm cắt không được đi vào feature của test sample.

**Nghiệm thu**

- [ ] `splits.yaml` ghi rõ protocol, fold, bearing IDs và seed.
- [ ] Assert giao của tập bearing giữa train/validation/test bằng rỗng.
- [ ] Không có scaler/statistic/FPT threshold nào được fit từ validation hoặc test.
- [ ] Test tự động thất bại nếu một window của cùng bearing xuất hiện ở hai split.

### Task 3: Làm Sạch Và Tiền Xử Lý Vibration

Pipeline phải hỗ trợ ba cấu hình để phục vụ ablation:

1. `raw_demean`: bỏ mean theo snapshot/kênh, giữ nguyên 2.560 mẫu.
2. `bandpass_full`: lọc trên tín hiệu 25,6 kHz bằng Butterworth SOS và `sosfiltfilt`; cutoff là tham số cấu hình, không mặc định tuyên bố 100 Hz-10 kHz là tối ưu.
3. `resample_1024`: lọc anti-alias rồi resample 2.560 -> 1.024 bằng `resample_poly` hoặc phương pháp tương đương.

Lưu ý bắt buộc: sau khi 0,1 giây được đưa về 1.024 mẫu, sampling rate hiệu dụng là 10,24 kHz và Nyquist chỉ còn 5,12 kHz. Vì vậy không thể vừa downsample còn 1.024 điểm vừa tuyên bố bảo toàn phổ đến 10 kHz.

Hai trục X/Y phải được xử lý giống nhau nhưng không trộn kênh. Không chuẩn hóa từng snapshot về unit variance vì thao tác đó có thể xóa thông tin tăng biên độ do suy thoái.

**Nghiệm thu**

- [ ] Output có shape `[2, 2560]` hoặc `[2, 1024]` đúng theo config.
- [ ] Không có NaN/Inf sau filter/resample.
- [ ] Có biểu đồ/metric so sánh trước-sau cho tối thiểu một bearing mỗi condition: RMS, peak, FFT/PSD và aliasing check.
- [ ] Filter dùng SOS ổn định số; test xác nhận shape, dtype và tính deterministic.
- [ ] Cutoff, order, sampling rate và resampling method được lưu trong artifact cấu hình.

### Task 4: Parse Và Đồng Bộ Temperature

Không upsample một scalar thành `[1, 64]`. Thực hiện theo timestamp:

- Parse toàn bộ temperature samples thành chuỗi `(timestamp, value)` cho từng bearing.
- Loại/đánh dấu giá trị ngoài miền vật lý bằng quy tắc cấu hình; không tự động xóa chỉ vì nằm ngoài khoảng giả định nếu chưa có bằng chứng sensor lỗi.
- Tại mỗi vibration acquisition, ghép temperature bằng interpolation tuyến tính hoặc nearest-neighbor có `max_alignment_gap_s`.
- Lưu `alignment_delta_s` và `temp_missing_mask`.

Hỗ trợ hai representation:

1. `temp_features` - vector causal gồm current temperature, mean, std, min, max, slope và time-since-last-valid trong một history horizon xác định.
2. `temp_history_64` - lấy **64 điểm nhiệt độ lịch sử có thật hoặc được nội suy trên khoảng lịch sử kết thúc tại acquisition hiện tại**; không dùng dữ liệu tương lai.

**Nghiệm thu**

- [ ] Alignment chỉ dùng temperature tại hoặc trước thời điểm dự đoán, trừ khi config nghiên cứu cho phép interpolation hai phía và được ghi rõ.
- [ ] Tỷ lệ missing, phân bố `alignment_delta_s` và số điểm bị loại được báo cáo theo bearing.
- [ ] Không có leakage từ temperature sau thời điểm cuối window.
- [ ] Output là `[F_temp]` hoặc `[1, 64]` theo config và có mask đi kèm khi cần.

### Task 5: Tạo Nhãn RUL

**Nhãn chính phải là RUL tuyệt đối**, không chuẩn hóa theo tổng tuổi thọ riêng của từng test bearing vì tổng tuổi thọ đó không biết tại lúc inference.

**Learning bearings**

- Với bearing run-to-failure, dùng timestamp/index của điểm cuối hợp lệ làm `failure_time` theo quy ước dataset.
- `rul_seconds(t) = max(0, failure_time - t)`.
- `rul_cycles` phải có định nghĩa duy nhất trong config, ví dụ số khoảng acquisition 10 giây còn lại.
- Tùy chọn `max_files` chỉ giới hạn dữ liệu được load/debug; nhãn vẫn phải lấy từ manifest vòng đời đầy đủ.

**Truncated test bearings**

- Tại điểm cắt: dùng official ground-truth RUL hoặc giá trị suy ra từ `Full_Test_Set` theo protocol đã phê duyệt.
- Với window kết thúc trước điểm cắt: `RUL(t) = RUL_at_cut + (t_cut - t)`.
- Không dùng signal sau điểm cắt để tạo input hoặc feature.

**Piecewise/FPT label**

- Không dùng `kurtosis <= 3` hoặc “RMS ổn định” làm định nghĩa mặc định.
- Piecewise RUL/FPT là một experiment tùy chọn sau baseline linear RUL.
- Nếu triển khai, FPT detector phải được định nghĩa bằng thuật toán cụ thể, fit threshold trên train-only, có sensitivity analysis và lưu cả nhãn linear lẫn piecewise để ablation.

**Nghiệm thu**

- [ ] RUL không âm và đơn điệu không tăng theo thời gian trong từng bearing.
- [ ] RUL cuối learning bearing bằng 0 theo tolerance đã định nghĩa.
- [ ] Debug subset không làm thay đổi nhãn của cùng acquisition.
- [ ] Unit test kiểm tra công thức test RUL tại điểm cắt và các thời điểm trước đó.
- [ ] Lưu đồng thời giá trị tuyệt đối và target transform/cap nếu mô hình cần; mọi cap phải được suy ra từ train-only.

### Task 6: Fit Và Áp Dụng Normalization

- Fit scaler sau khi đã khóa split và chỉ trên train bearings của fold hiện tại.
- Tính statistic riêng cho Vib-X, Vib-Y và từng temperature feature.
- Mặc định dùng global train-channel z-score hoặc robust scaling. Không fit scaler riêng cho từng bearing validation/test.
- Có epsilon cho kênh variance gần 0 và optional clipping theo quantile học từ train-only.
- Lưu `scaler.json`/`scaler.npz` cùng hash của split và preprocessing config.

**Nghiệm thu**

- [ ] Scaler artifact chứa count, mean/std hoặc median/IQR, epsilon và nguồn split.
- [ ] Transform/inverse-transform có round-trip test cho target nếu target được scale.
- [ ] Thay đổi test data không làm thay đổi scaler hash.
- [ ] Báo cáo phân phối train/validation/test sau transform nhưng không tái-fit.

### Task 7: Sliding Window Theo Chuỗi Acquisition

- Mặc định `sequence_length = 32` acquisitions.
- Mặc định `stride_train = 16`; validation/test dùng stride theo protocol đánh giá.
- Khoảng cách giữa tâm acquisition đầu và cuối của window 32 bước cách nhau 10 giây là khoảng 310 giây; không mô tả đây là 1,28 giây.
- Window chỉ được tạo sau khi từng acquisition đã có vibration, temperature/mask, timestamp và RUL.
- Target của window là RUL tại acquisition cuối.

**Shape chuẩn đề xuất**

```text
vibration:       [T, 2, L_vib]       # L_vib = 2560 hoặc 1024
temperature:     [T, F_temp]          # temp_features
# hoặc
temperature:     [T, 1, 64]           # temp_history_64
temperature_mask:[T, ...]
rul:             [1]
metadata:        bearing_id, condition_id, end_timestamp, end_index
```

**Nghiệm thu**

- [ ] Không có window vắt qua hai bearing hoặc hai split.
- [ ] Timestamp trong window tăng và target đúng với timestep cuối.
- [ ] Công thức số window được test cho nhiều `T/stride`.
- [ ] Mỗi sample giữ metadata để truy vết ngược về file thô.

### Task 8: Cache, PyTorch Dataset Và DataLoader

Không bắt buộc gom toàn bộ dữ liệu vào một file `.npy` khổng lồ. Ưu tiên shard theo bearing để dễ kiểm tra và tránh vượt RAM.

**Artifact tối thiểu**

```text
artifacts/<pipeline_version>/
  manifest.parquet
  data_quality_report.json
  splits.yaml
  preprocessing.yaml
  scaler.json
  label_report.json
  shards/<split>/<bearing_id>.npz
  cache_index.parquet
```

`PronostiaDataset` phải trả về dictionary thống nhất:

```python
{
    "vibration": FloatTensor,       # [T, 2, L_vib]
    "temperature": FloatTensor,     # [T, F] hoặc [T, 1, 64]
    "temperature_mask": BoolTensor,
    "rul": FloatTensor,             # [1]
    "bearing_id": str,
    "condition_id": str,
    "end_index": int,
}
```

DataLoader train được shuffle **giữa các window đã thuộc train split**; validation/test không shuffle. `pin_memory`, `num_workers` và `persistent_workers` là config theo môi trường.

**Nghiệm thu**

- [ ] Batch có shape đúng config và metadata vẫn ánh xạ đúng từng sample.
- [ ] Chạy được full pass qua từng split mà không NaN/Inf/crash.
- [ ] Cache invalidates khi raw checksum, split, scaler hoặc preprocessing config thay đổi.
- [ ] Benchmark ghi median/P95 load time, samples/s, RAM và cấu hình máy. Không dùng mốc `<10 ms/batch` làm hard gate nếu chưa cố định phần cứng.
- [ ] Hai lần chạy cùng config/seed tạo manifest, split và label giống nhau.

### Task 9: Data Validation Report

Sinh báo cáo tự động trước khi bàn giao:

- Số bearing/file/acquisition/window theo subset, condition và split.
- Thời lượng quan sát, RUL min/max và số window theo bearing.
- Tỷ lệ file lỗi/quarantine, NaN/Inf và temperature missing.
- Histogram/quantile của Vib-X RMS, Vib-Y RMS và temperature.
- Kiểm tra RUL monotonicity.
- Kiểm tra overlap giữa split.
- Kiểm tra alignment delta.
- Ví dụ reconstruction/tracing từ một batch về raw filenames.

Pipeline không được đánh dấu hoàn thành nếu validation report có lỗi mức `ERROR`.

---

## 6. Cấu Trúc Mã Nguồn Bàn Giao

```text
src/
  data/
    schemas.py
    inventory.py
    parsers.py
    alignment.py
    vibration.py
    temperature.py
    labeling.py
    splitting.py
    windowing.py
    scaling.py
    cache.py
    dataset.py
    validation.py
configs/
  data/pronostia.yaml
  split/development_cv.yaml
  split/official_benchmark.yaml
tests/
  unit/
  integration/
scripts/
  build_manifest.py
  build_cache.py
  validate_data.py
```

Không để toàn bộ pipeline trong một notebook. Notebook chỉ dùng cho EDA và trực quan hóa; logic sản xuất phải nằm trong module có test.

---

## 7. Definition Of Done Cho Phần 1

- [ ] Pipeline chạy từ data root sạch đến cache bằng một command có config.
- [ ] Manifest và quality report bao phủ toàn bộ dataset.
- [ ] Development CV và official benchmark split không leakage.
- [ ] Vibration hai trục được giữ; downsampling có anti-alias và được kiểm chứng.
- [ ] Temperature được đồng bộ bằng timestamp và có missing mask.
- [ ] Linear absolute RUL đúng, đơn điệu và không đổi khi dùng debug subset.
- [ ] Scaler fit train-only và có artifact tái sử dụng.
- [ ] Window không vượt ranh giới bearing/split; target thuộc timestep cuối.
- [ ] Dataset/DataLoader chạy full pass ở mọi split.
- [ ] Unit và integration tests pass.
- [ ] Có README hướng dẫn command, cấu hình, output và cách tái lập.
- [ ] Có validation report không còn lỗi mức `ERROR`.

---

## 8. Ngoài Phạm Vi Phần 1

Các nội dung sau thuộc phần modeling/evaluation, không phải tiêu chí hoàn thành của nhân sự Phần 1:

- Cài đặt CNN, LSTM/GRU, Transformer hoặc cross-modal attention.
- Stage-aware/asymmetric training loss.
- Hyperparameter search của mô hình.
- Kết quả MAE/RMSE/PHM score và ablation study.
- Các con số mục tiêu trong tài liệu khách hàng.

Phần 1 chỉ chịu trách nhiệm cung cấp dữ liệu đúng, truy vết được, không leakage và đủ ổn định để các mô hình phía sau được so sánh công bằng.

---

## 9. Tài Liệu Tham Chiếu Nền

- IEEE PHM 2012 Prognostic Challenge, `IEEEPHM2012-Challenge-Details.pdf` đi kèm dataset.
- Nectoux et al., *PRONOSTIA: An Experimental Platform for Bearings Accelerated Degradation Tests*, IEEE PHM 2012.

Mọi khác biệt giữa tài liệu này và file CSV thực tế phải được ghi vào `data_quality_report.json` và ưu tiên bằng chứng từ dataset/benchmark gốc thay vì giả định trong notebook mẫu.
