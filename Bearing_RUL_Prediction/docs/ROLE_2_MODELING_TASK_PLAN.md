# Kế Hoạch Công Việc Role 2: Modeling Và Multi-Modal Fusion

**Dự án:** Bearing Remaining Useful Life Prediction
**Vai trò:** Role 2 - Deep Learning Modeling
**Phụ thuộc đầu vào:** Data contract trong `DATA_PREPROCESSING_PLAN.md`
**Phối hợp:** Role 1 cung cấp dữ liệu; Role 3 cung cấp trainer, loss, metrics và evaluation
**Trạng thái:** Có thể triển khai ngay bằng synthetic data, không chờ pipeline dữ liệu thật hoàn thành

---

## 1. Mục Tiêu

Role 2 chịu trách nhiệm xây dựng, kiểm thử và bàn giao các kiến trúc dự đoán RUL từ đơn giản đến mô hình đa phương thức đề xuất.

Đầu ra phải đáp ứng:

1. Model độc lập với cách đọc CSV và đường dẫn dataset.
2. Model nhận batch đúng data contract và trả về RUL có shape `[B, 1]`.
3. Có baseline để chứng minh giá trị của từng thành phần mới.
4. Mỗi module có unit test về shape, gradient và numerical stability.
5. Mọi kiến trúc được tạo từ config, không hard-code batch size hoặc sequence length.
6. Có thể tích hợp với trainer của Role 3 mà không sửa logic nội bộ.

---

## 2. Phạm Vi Trách Nhiệm

### Role 2 thực hiện

- Định nghĩa model input/output contract.
- Xây vibration encoder.
- Xây temperature encoder.
- Xây các baseline đơn phương thức và fusion đơn giản.
- Xây multi-scale cross-modal fusion.
- Xây temporal encoder và regression head.
- Viết model factory và model configuration.
- Viết unit tests, smoke tests và overfit-one-batch test hỗ trợ tích hợp.
- Báo cáo parameter count, tensor shape và chi phí forward pass.

### Không thuộc Role 2

- Đọc, làm sạch, đồng bộ hoặc cache CSV thô.
- Chia train/validation/test và fit scaler.
- Định nghĩa ground-truth RUL.
- Training loop, checkpoint, early stopping hoặc scheduler.
- Cài đặt metric/PHM score và báo cáo kết quả thực nghiệm.
- Stage-aware/asymmetric loss; phần này thuộc Role 3 để quản lý chung training objective.
- Tuyên bố các con số RMSE, PHM score hoặc mức cải thiện khi chưa có kết quả chạy thật.

---

## 3. Input Và Output Contract

Model phải hỗ trợ batch tối thiểu:

```python
batch = {
    "vibration": FloatTensor,       # [B, T, 2, L_vib]
    "temperature": FloatTensor,     # [B, T, F_temp] hoặc [B, T, 1, 64]
    "temperature_mask": BoolTensor, # cùng temporal layout với temperature
    "rul": FloatTensor,             # [B, 1], chỉ dùng bên ngoài model
}
```

Trong đó:

- `B`: batch size bất kỳ.
- `T`: sequence length bất kỳ, mặc định 32.
- `L_vib`: 1.024 hoặc 2.560 theo preprocessing config.
- `F_temp`: số temperature features; không được giả định cố định nếu config chưa khóa.
- Model không được sử dụng `rul` trong `forward`.

Output chuẩn:

```python
output = {
    "rul_pred": FloatTensor,         # [B, 1], không âm
    "aux": {                         # optional khi return_aux=True
        "scale_weights": ...,
        "cross_attention": ...,
        "temporal_weights": ...,
    },
}
```

`rul_pred` phải có đơn vị/scale giống target do Role 3 truyền vào. Regression head mặc định dùng `Softplus` hoặc cơ chế tương đương để tránh RUL âm; phải cho phép tắt qua config để chạy ablation.

---

## 4. Synthetic Data Để Làm Song Song

Trong khi chờ Role 1, tạo fixture dùng chung:

```python
def make_synthetic_batch(
    batch_size=4,
    sequence_length=32,
    vibration_length=1024,
    temperature_features=7,
):
    return {
        "vibration": torch.randn(
            batch_size, sequence_length, 2, vibration_length
        ),
        "temperature": torch.randn(
            batch_size, sequence_length, temperature_features
        ),
        "temperature_mask": torch.ones(
            batch_size, sequence_length, temperature_features,
            dtype=torch.bool,
        ),
        "rul": torch.rand(batch_size, 1) * 1000,
    }
```

Fixture phải hỗ trợ thêm các case:

- Batch size bằng 1.
- Sequence ngắn hơn 32.
- Vibration length bằng 1.024 và 2.560.
- Temperature có missing mask.
- Toàn bộ temperature của một timestep bị thiếu.
- Giá trị đầu vào lớn/nhỏ bất thường nhưng hữu hạn.

---

## 5. Thứ Tự Task Và Điều Kiện Nghiệm Thu

### Task 1: Model API, Base Classes Và Configuration

**Công việc**

- Tạo `ModelConfig` bằng dataclass hoặc cấu hình tương đương.
- Định nghĩa interface thống nhất cho tất cả model.
- Tạo model registry/factory theo tên.
- Hỗ trợ `return_aux=False/True`.
- Thêm validation config: số head chia hết cho `d_model`, dimension dương và dropout hợp lệ.

**Nghiệm thu**

- [X] Tất cả model được khởi tạo bằng config.
- [X] Không hard-code `B`, `T` hoặc device.
- [X] Forward chạy trên CPU và CUDA nếu môi trường có GPU.
- [X] Model factory báo lỗi rõ ràng với tên/config không hợp lệ.

### Task 2: Vibration Encoder

Xử lý từng acquisition độc lập trước khi temporal modeling:

```text
[B, T, 2, L]
  -> reshape [B*T, 2, L]
  -> shared 1D CNN blocks
  -> multi-scale pooled embeddings
  -> restore [B, T, D]
```

Yêu cầu:

- Giữ hai trục X/Y làm input channel.
- Dùng `Conv1d + normalization + activation + pooling` có cấu hình.
- Trả về ba scale `early`, `mid`, `late`.
- Dùng adaptive pooling để hỗ trợ cả `L=1024` và `L=2560`.
- Không flatten toàn bộ waveform thành fully connected vector lớn.

**Nghiệm thu**

- [X] Cả ba scale có shape `[B, T, D_scale]`.
- [X] Cùng model chạy được với hai vibration lengths.
- [X] Gradient đi từ output về tất cả convolution blocks.
- [X] Parameter count và receptive field được ghi lại.

### Task 3: Temperature Encoder

Hỗ trợ hai representation:

1. Feature vector: `[B, T, F_temp]`.
2. Temperature history: `[B, T, 1, 64]`.

Thiết kế:

- Feature vector dùng MLP/gated MLP.
- History 64 điểm dùng lightweight `Conv1d` hoặc temporal MLP.
- Missing mask phải được dùng trong encoder hoặc trước encoder; không được mặc định biến missing thành measurement bằng 0 mà không có mask embedding.
- Output trả về `early`, `mid`, `late` theo dimension cấu hình.

**Nghiệm thu**

- [X] Hai input representation đều được test.
- [X] Missing temperature không tạo NaN.
- [X] Output có shape `[B, T, D_scale]`.
- [X] Không đọc temperature ở timestep tương lai ngoài batch được cung cấp.

### Task 4: Baseline Models

Cài đặt tối thiểu bốn baseline:

1. `VibrationCNNRegressor`.
2. `TemperatureRegressor`.
3. `LateConcatFusionRegressor`.
4. `CNNTemporalRegressor` - concat hai modality rồi GRU/LSTM.

Mục đích là tạo mốc so sánh cho multi-scale attention. Baseline phải dùng chung encoder/head khi phù hợp để ablation công bằng.

**Nghiệm thu**

- [X] Mỗi baseline forward ra `[B, 1]`.
- [X] Có config bật/tắt từng modality.
- [X] Cùng một batch có thể chạy qua toàn bộ baseline.
- [X] Parameter count được xuất tự động.

### Task 5: Cross-Modal Attention Block

Attention phải chạy trên **temporal tokens trong từng sample**, không attention qua chiều batch.

Chuẩn hóa dimension trước attention:

```text
vibration scale [B, T, D_v]
temperature scale [B, T, D_t]
  -> linear projection về d_model
  -> cross-attention trong từng sample
  -> residual + normalization
  -> fused scale [B, T, d_model]
```

Khuyến nghị triển khai hai chiều:

- Vibration query, temperature key/value.
- Temperature query, vibration key/value.
- Ghép/gate hai kết quả thành fused representation.

Temperature mask phải được chuyển đúng thành attention mask. Không tạo ma trận attention giữa các phần tử của batch.

**Nghiệm thu**

- [X] Thay đổi sample thứ hai không làm thay đổi output sample thứ nhất ở `eval()`.
- [X] Attention weights có shape chứa riêng `B`, head, query time và key time.
- [X] Masked temperature token không nhận attention probability đáng kể.
- [X] Có residual, normalization và dropout theo config.
- [X] Forward/backward ổn định với batch size 1.

### Task 6: Multi-Scale Fusion

Không cộng trực tiếp tensor 64/128/256 chiều. Mỗi scale phải được project về cùng `d_model` trước khi fusion.

Hỗ trợ tối thiểu hai mode:

1. `concat_project`: concatenate ba scale rồi linear projection.
2. `weighted_sum`: project từng scale về cùng dimension, sau đó dùng softmax scale weights.

```text
early -> Project(d_model) --┐
mid   -> Project(d_model) --+-> scale fusion -> [B, T, d_model]
late  -> Project(d_model) --┘
```

Scale weights có thể là:

- Global learnable weights; hoặc
- Dynamic weights theo sample/timestep.

Global weights được ưu tiên cho phiên bản đầu vì dễ kiểm tra và giải thích.

**Nghiệm thu**

- [X] Scale weights sau softmax có tổng bằng 1 trong tolerance.
- [X] Ba scale có gradient khác 0 trong backward test.
- [X] `concat_project` và `weighted_sum` dùng cùng output dimension.
- [X] Aux output trả scale weights khi được yêu cầu.

### Task 7: Temporal Modeling

Triển khai theo hai cấp:

**MVP temporal encoder**

- GRU hoặc LSTM nhận `[B, T, d_model]`.
- Temporal attention pooling hoặc lấy hidden state cuối.
- Đây là model chính để integration sớm.

**Dual-path temporal encoder**

- Fast path ưu tiên vibration/fused transient representation.
- Slow path ưu tiên temperature/trend representation.
- Hai path phải nhận input có ý nghĩa khác nhau; không chỉ tạo hai LSTM khác size cùng nhận một tensor giống hệt nhau.
- Project hai hidden state về cùng dimension trước gated/weighted fusion.

```text
fast features -> Fast GRU/LSTM -> project --┐
                                             +-> gated temporal fusion
slow features -> Slow GRU/LSTM -> project --┘
```

Không mặc định dùng bidirectional encoder nếu mục tiêu triển khai yêu cầu causal online inference. Cần có config `bidirectional` và ghi rõ protocol.

**Nghiệm thu**

- [X] Hỗ trợ sequence length thay đổi.
- [X] Fast/slow output được project về dimension tương thích.
- [X] Temporal weights hợp lệ và được trả qua aux output.
- [X] Causal mode có test đảm bảo thay đổi timestep tương lai không ảnh hưởng output ở timestep trước nếu model xuất sequence.

### Task 8: Regression Head Và Full Proposed Model

Cấu trúc mặc định:

```text
temporal embedding
  -> Linear
  -> activation
  -> dropout
  -> Linear
  -> Softplus
  -> RUL [B, 1]
```

Full model dự kiến:

```text
Vibration multi-scale CNN -------┐
                                  -> multi-scale cross-modal attention
Temperature multi-scale encoder -┘
                                  -> scale fusion
                                  -> dual-path temporal encoder
                                  -> regression head
                                  -> RUL
```

**Nghiệm thu**

- [X] Output luôn hữu hạn và đúng shape.
- [X] Với non-negative head, output không âm.
- [X] `return_aux=False` không giữ tensor attention lớn không cần thiết.
- [X] Full model backward qua tất cả module.
- [X] Có thể save/load `state_dict` và cho output giống nhau trong `eval()`.

### Task 9: Model Tests Và Integration Readiness

Bộ test tối thiểu:

- Shape test cho tất cả model và module.
- Batch-independence test cho attention.
- Mask behavior test.
- Gradient-flow test.
- NaN/Inf test.
- Deterministic eval test.
- Save/load round-trip test.
- CPU/GPU parity trong tolerance nếu có GPU.
- Overfit-one-batch smoke test phối hợp với Role 3.

Overfit-one-batch chỉ là kiểm tra wiring: loss phải giảm mạnh trên một batch nhỏ. Đây không phải bằng chứng model tổng quát hóa.

**Nghiệm thu**

- [X] Toàn bộ unit tests pass.
- [X] Model + dummy trainer chạy ít nhất một forward/backward/update cycle.
- [X] Không có unused trainable module trong full model.
- [X] Có báo cáo peak memory và forward latency trên cấu hình máy được dùng.

---

## 6. Danh Sách Model Chuẩn

| Model key                        | Mục đích                            |
| -------------------------------- | -------------------------------------- |
| `vib_cnn`                      | Baseline vibration-only                |
| `temp_mlp`                     | Baseline temperature-only              |
| `late_concat`                  | Baseline fusion đơn giản            |
| `cnn_gru`                      | Baseline temporal đa modality         |
| `single_scale_cross_attention` | Đo riêng đóng góp cross-attention |
| `multi_scale_fusion`           | Đo riêng đóng góp multi-scale     |
| `multi_scale_cross_attention`  | Multi-scale + cross-modal              |
| `full_dual_path`               | Mô hình đề xuất đầy đủ        |

Tên model phải ổn định để Role 3 dùng trong experiment config.

---

## 7. Cấu Trúc Mã Nguồn Bàn Giao

```text
src/
  models/
    config.py
    factory.py
    common.py
    vibration_encoder.py
    temperature_encoder.py
    cross_attention.py
    scale_fusion.py
    temporal.py
    regression.py
    baselines.py
    multimodal_rul.py
tests/
  models/
    conftest.py
    test_vibration_encoder.py
    test_temperature_encoder.py
    test_cross_attention.py
    test_scale_fusion.py
    test_temporal.py
    test_baselines.py
    test_multimodal_rul.py
configs/
  model/
    vib_cnn.yaml
    temp_mlp.yaml
    late_concat.yaml
    cnn_gru.yaml
    full_dual_path.yaml
```

Không triển khai model chính chỉ trong notebook. Notebook có thể dùng để minh họa tensor/attention nhưng source of truth phải nằm trong `src/models`.

---

## 8. Kế Hoạch Mốc Bàn Giao

### Milestone M1: Contract Và Baselines

- Model config/factory.
- Synthetic fixtures.
- Vibration-only, temperature-only và concat baselines.
- Unit tests cơ bản.

**Gate:** Role 3 gọi được model bằng một API thống nhất.

### Milestone M2: Fusion Và Temporal MVP

- Cross-modal attention đúng chiều.
- Multi-scale projection/fusion.
- GRU/LSTM temporal MVP.
- Aux outputs cho attention/scale weights.

**Gate:** Full forward/backward chạy bằng synthetic batch.

### Milestone M3: Full Dual-Path Model

- Fast/slow inputs có ý nghĩa riêng.
- Gated temporal fusion.
- Regression head không âm.
- Save/load và overfit-one-batch smoke test.

**Gate:** Model sẵn sàng nhận DataLoader thật của Role 1.

### Milestone M4: Real-Data Integration

- Chạy shape/data validation trên một batch thật.
- Điều chỉnh adapter nếu representation temperature đã được hai bên thống nhất thay đổi.
- Không thay đổi data split hoặc label của Role 1 trong model code.
- Bàn giao parameter/latency/memory report cho Role 3.

**Gate:** Role 3 chạy được một experiment end-to-end mà không sửa source model.

---

## 9. Quy Tắc Phối Hợp Với Role 1 Và Role 3

### Với Role 1

- Khóa sớm các trường: tensor shape, dtype, mask semantics, target unit và metadata.
- Nếu data contract thay đổi, cập nhật version thay vì sửa ngầm.
- Role 2 không truy cập raw CSV để tự tạo pipeline song song.

### Với Role 3

- Role 2 cung cấp `build_model(config)` và output dictionary thống nhất.
- Role 3 quyết định loss, optimizer, scheduler, metric và checkpoint monitor.
- Attention/scale weights chỉ là diagnostic output; trainer không được phụ thuộc vào chúng trừ experiment được khai báo rõ.

### Integration gate chung

```text
Role 1: real batch ───────┐
                          ├── Role 3 trainer/evaluator
Role 2: model factory ────┘
```

Integration test phải xác nhận:

1. Batch thật chạy được qua model không cần reshape thủ công bên ngoài.
2. Prediction và target cùng shape/scale.
3. Missing temperature được xử lý đúng.
4. Một optimizer step chạy không NaN.
5. Checkpoint save/load thành công.

---

## 10. Definition Of Done Cho Role 2

- [X] Có đủ bốn baseline và full proposed model.
- [X] Tất cả model dùng chung input/output API.
- [X] Hỗ trợ `L_vib=1024` và `L_vib=2560` bằng adaptive operations.
- [X] Hỗ trợ temperature features và temperature history theo config.
- [X] Cross-attention không trộn sample qua batch.
- [X] Multi-scale tensors được project về cùng dimension trước weighted sum.
- [X] Fast/slow temporal paths có input khác nhau và dimension fusion hợp lệ.
- [X] Output RUL đúng shape, hữu hạn và mặc định không âm.
- [X] Unit tests, gradient tests, mask tests và save/load tests pass.
- [X] Synthetic forward/backward/update chạy thành công.
- [X] Real batch của Role 1 chạy qua model không cần sửa contract.
- [X] Role 3 có thể khởi tạo mọi model bằng model key/config.
- [X] Không có kết quả giả hoặc metric placeholder được ghi như kết quả thực nghiệm.

Role 2 hoàn thành khi model library đã sẵn sàng cho các experiment có kiểm soát; không cần chờ toàn bộ huấn luyện và bảng kết quả cuối cùng hoàn tất.
