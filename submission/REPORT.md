# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thanh Huy  **MSSV**: 2A202601802  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 15360 MiB (driver 580.82.07, CUDA 13.0), Colab free

> Mọi con số dưới đây lấy trực tiếp từ `results/`. Stack của lần chạy được ghi trong
> `results/ENVIRONMENT.txt`: torch 2.11.0+cu128 · transformers 5.15.0 · trl 1.10.0 ·
> peft 0.20.0 · accelerate 1.14.0 · datasets 5.0.1 · bitsandbytes 0.50.1 · torchao
> 0.18.0. Commit của lần chạy: `20c31045`.

---

## 1. Setup

| | |
|---|---|
| Dataset | corpus mặc định — 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (mặc định của tier) — p95 đo được là **98** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **30 step** (batch 1 × grad_accum 16 = batch hiệu dụng 16, dưới ngưỡng 32 của §10.4) |
| Chế độ eval | đầy đủ 50 mẫu target + 15 mẫu regression, `EVAL_LIMIT` không đặt |

**Template có giữ khối `<think>` không?** **Có** — `results/template_check.json` cho
`verdict: "reasoning preserved — safe to train on traces"`, với `open_tag_present` và
`body_present` đều `true`. Chuỗi render giữ nguyên `<think>\nbuoc 1: kiem tra...\n</think>`,
nghĩa là template không nuốt nội dung suy luận do người viết cung cấp. Điều này *cho phép*
huấn luyện trên trace, nhưng corpus mặc định không có trace nào — cả 250 câu trả lời đều là
JSON trần. Hệ quả xuất hiện lại ở mục 5 dưới dạng `valid_trace_rate = 0.0`.

**Về `max_length`:** tôi giữ 1024 của tier chứ không hạ xuống 256 như p95 gợi ý, và đây là
một lựa chọn có đánh đổi rõ ràng. p95 = 98 token, max tuyệt đối = 101, nên **không mẫu nào
bị cắt** ở cả hai giá trị — 1024 không mua thêm tính đúng đắn, nó chỉ tiêu thêm bộ nhớ và
thời gian cho phần đệm. Lý do giữ: mọi số VRAM và thời gian trong report này so sánh được
với `docs/MEASURED-T4-2026-08-20.md`, vốn cũng chạy ở 1024; hạ xuống 256 sẽ làm bốn run đối
chứng không còn đối chiếu được với mốc đã công bố. Nếu chạy lại vì mục đích sản xuất chứ
không phải để so sánh, 256 là lựa chọn đúng.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **không** được tính loss:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

```

Hai điều đáng chú ý. Thứ nhất, `supervised_fraction = 0.4149` — xa ngưỡng cảnh báo 0.95,
nên loss không hề tính trên prompt. Thứ hai, ranh giới rơi vào **giữa** khối think: thẻ mở
`<think>` nằm trong phần bị mask, còn thẻ đóng `</think>` lại nằm trong phần được tính loss.
Đó không phải lỗi — Qwen3.5 phát ra `<think>\n\n</think>\n\n` ngay cả khi câu trả lời không
có suy luận nào, và thẻ mở thuộc về generation prompt. Mô hình vẫn học được rằng "kết thúc
suy luận rồi mới xuất JSON", và quan trọng hơn là `<|im_end|>` — tín hiệu dừng — nằm trong
loss. Thiếu nó thì mô hình sẽ không bao giờ học được lúc nào cần im lặng.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3193.1 |
| (b) base + optimized prompt | 0.765 | 0.7578 | **1.000** | **1011.8** |
| (c) LoRA fine-tune | **0.970** | 0.5889 | **1.000** | 1461.3 |

**(b) có thật sự mạnh hơn (a) không?** **Có, và cách biệt rất lớn.** (a) đạt `target 0.000`
và `format 0.000` — không phải nó phân loại sai, mà nó **không hề trả về JSON**: prompt ngây
thơ "Phân loại ticket sau." không nói cho model biết cần 4 khoá nào hay có những nhãn nào,
nên nó viết văn xuôi cho tới khi cụt token. (b) đạt 0.765 với JSON hợp lệ 100%.

Tôi **không sửa** `OPTIMIZED_PROMPT`. `make verify` xác nhận SHA `719e74d3b6232053` khớp bản
gốc — cổng này tồn tại vì làm yếu (b) đi là cách dễ nhất để fine-tune trông có vẻ thắng.

Một quan sát ngoài lề nhưng đáng tiền: **(b) nhanh hơn (a) 3,2 lần** (1011.8 ms so với
3193.1 ms). Prompt tốt không chỉ chính xác hơn, nó còn rẻ hơn — vì model không phải sinh ra
hàng trăm token thừa. Đây là F-12 trong `SIMULATION-FINDINGS.md`, tái lập đúng trên lần chạy
này.

---

## 4. Giải phẫu cấu hình sai (NB4)

Bốn run, **cùng 30 step**, **cùng ngân sách tham số** (`make verify` kiểm tự động cả hai).

| Run | vị trí | modules | r | trainable | LR | train loss (NB4) | **target (NB5)** | format | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 12 | 16 | 32,464,896 | 1e-4 | 0.6267 | **0.970** | 1.000 | 1004.5 | 12.01 |
| `attn_only` | q,v | 2 | **283** | 32,456,704 | 1e-4 | **0.5378** | **0.970** | 1.000 | 913.8 | 12.02 |
| `wrong_lr` | text-linear | 12 | 16 | 32,464,896 | **1e-5** | **1.5704** | **0.000** | 0.000 | 995.9 | 12.01 |
| `qlora` | text-linear | 12 | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1.000 | 1047.3 | **7.09** |

Ngân sách khớp ở mức 32,456,704 so với 32,464,896 — lệch **0,025%**, thoải mái dưới ngưỡng
5% của cổng công bằng.

### 4.1 — `attn_only` so với `correct`: hoà trên target, nhưng train loss xếp ngược

Trên tập target hai run **hoà tuyệt đối**: 0.970 và 0.970, format 1.000 cả hai. Nhưng theo
`final_loss` thì `attn_only` **thắng** — 0.5378 so với 0.6267, thấp hơn 14%. Nếu chấm bằng
cột loss như bản lab cũ vẫn làm, tôi đã kết luận rằng đặt adapter chỉ ở `q,v` là cấu hình
tốt hơn, tức là đúng cái deck gọi là **Lỗi #1**. Hai cột cho hai thứ tự khác nhau, và đây là
kết quả đáng giá nhất tôi đo được trong lab: train loss không chỉ là bằng chứng yếu, ở đây
nó là bằng chứng **sai hướng**.

Về *vị trí so với rank*: để bằng ngân sách tham số với `correct` ở r=16 trải trên 12 loại
module, `attn_only` phải nâng rank lên **283** — gấp 17,7 lần — mà chỉ gắn vào 2 loại module.
Đổi 17,7 lần rank lấy 1/6 độ phủ module, kết quả là **hoà**. Diễn giải trung thực: trên bài
toán này, ở ngân sách này, tôi **không chứng minh được** vị trí là đòn bẩy — tôi chỉ chứng
minh được rank không bù lại được gì. Cả hai đều chạm 0.970 trên thang đo, và 6 lỗi còn lại
của `correct` đều là cùng một loại lỗi hệ thống (xem mục 6), nên nhiều khả năng tác vụ đã
đụng trần chứ không phải hai cấu hình thực sự tương đương. Muốn tách bạch thì cần một tác vụ
khó hơn, hoặc thang đo mịn hơn — đó là điều tôi sẽ làm nếu có thêm thời gian.

Một khác biệt **có** đo được nằm ở chỗ khác: `attn_only` suy luận nhanh hơn 1,6 lần
(918.8 ms so với 1461.3 ms), hợp lý vì nó chỉ chèn adapter vào 2 loại module thay vì 12, nên
chi phí mỗi token thấp hơn. Nếu hai cấu hình thật sự hoà về chất lượng thì đây là lập luận
để chọn `attn_only` — nhưng nó là kết luận về *chi phí phục vụ*, không phải về *đòn bẩy học*.

### 4.2 — `wrong_lr`: đổi đúng một con số, sập hoàn toàn

`wrong_lr` khác `correct` **duy nhất** ở learning rate: 1e-5 thay vì 1e-4 — thang của
full fine-tune áp cho LoRA, thấp hơn 10 lần. Loss dừng ở **1.5704**, gấp 2,5 lần `correct`
(0.6267), và trên thang đo tác vụ nó không đạt gì cả: `target 0.000`, **`format 0.000`**.
Con số format mới là thứ đáng sợ: sau 30 step nó chưa học nổi việc phát ra JSON, chứ chưa
nói tới việc phân loại đúng. Nó cũng là run chậm nhất khi sinh — 5193.5 ms/mẫu, gấp 3,6 lần
`correct` — vì giống baseline (a), nó viết lan man tới khi cụt token.

Nếu chỉ nhìn loss mà không biết LR, tôi sẽ kết luận sai theo một hướng rất cụ thể: "loss vẫn
đang giảm, mô hình cần thêm dữ liệu hoặc thêm epoch". Đó là chẩn đoán tự nhiên khi thấy một
đường loss cao nhưng chưa phẳng — và nó sẽ dẫn tới việc đốt thêm nhiều giờ GPU cho một tham
số sai lệch 10 lần mà chỉ cần sửa một ký tự. Đáng chú ý là ở run này, loss **xếp đúng**: sai
lệch 10 lần đủ lớn để lộ ra ngay cả trên một chỉ số tồi. Bài học không phải "loss vô dụng"
mà là "loss chỉ bắt được lỗi thô".

### 4.3 — `qlora`: tiết kiệm VRAM có thật, cái giá cũng có thật

`qlora` dùng **7.09 GB** so với 12.01 GB của `correct` — **ít hơn 41%**, đúng bằng con số
`docs/MEASURED-T4-2026-08-20.md` ghi nhận. Cái giá đo được có ba phần: target giảm từ 0.970
xuống **0.940** (−0.030), suy luận chậm hơn 23% (1797.3 ms so với 1461.3 ms), và train lâu
hơn một chút (1047.3 s so với 1004.5 s) — lượng tử hoá 4-bit không miễn phí ở cả hai chiều.

Số đo của tôi **ủng hộ có điều kiện** khuyến nghị "không dùng QLoRA cho dòng model này". Trên
chính T4 này, `correct` chỉ dùng 12.01 GB trong 14.6 GB khả dụng — VRAM chưa bao giờ là ràng
buộc, nên trả 3 điểm accuracy và 23% latency để lấy phần bộ nhớ không dùng tới là lỗ thuần.
Nhưng kết luận đó đến từ *phần cứng*, không phải từ QLoRA. Trên card 8 GB, nơi 12.01 GB đơn
giản là không chạy được, thì 0.940 là điểm hoàn toàn dùng được và lựa chọn duy nhất là giữa
QLoRA hoặc không có gì. Khuyến nghị của nhà cung cấp nên đọc là "đừng dùng khi bạn không cần",
chứ không phải "QLoRA hỏng".

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.169` (ngưỡng 0.020) · `valid_trace_rate = 0.00`

Bản fine-tune **thắng bài toán một cách áp đảo và thua bài kiểm tra tổng quát**. Trên nhóm
target nó đạt 0.970 so với 0.765 của baseline (b) — một prompt đã được tối ưu nghiêm túc,
không phải rơm. Đây là lần đầu tiên câu hỏi tiêu đề của lab được trả lời ở ngân sách đầy đủ
trên tier mặc định: `docs/MEASURED-T4-2026-08-20.md` ghi rõ ở mục "Still unrun anywhere"
rằng chưa ai đo được điều này, và lần đo rút gọn duy nhất trước đó cho 0.500 < 0.688, tức là
thua. Ở ngân sách đầy đủ, kết quả đảo chiều: fine-tune thắng +0.205.

Nhưng cổng hồi quy bốn nhóm bác bỏ nó, và bác bỏ vì đúng một lý do: năng lực tổng quát tụt
từ 0.7578 xuống 0.5889, tức −0.169 so với dung sai 0.020 — vượt ngưỡng **8,4 lần**. Đặt cạnh
lần chạy 0.8B đã ghi trong `SIMULATION-FINDINGS.md`, nơi regression sụp 0.644 → 0.067 gần như
xoá sạch, thì ở 4B nó giữ lại 78%. Model lớn hơn chống chịu tốt hơn — nhưng "tốt hơn" vẫn
không phải "đạt".

### 5.1 — Cái mất đi không phải kiến thức, mà là chỗ đặt câu trả lời

Con số −0.169 không nói được cơ chế, nên tôi đo tiếp từng câu một
(`results/regression_preds.json`, dựng lại đúng lời gọi của NB5 nên hai giá trị trung bình
khớp chính xác: base 0.7578, fine-tune 0.5889). Kết quả: **5/15 câu tệ hơn, 9 câu giữ
nguyên, 1 câu tốt hơn.**

Và cả 5 câu tệ hơn có chung một cơ chế — không phải quên, mà là **định dạng đầu ra bị bắt cóc**:

| # | Câu hỏi | base | FT | Fine-tune trả lời gì |
|---|---|---|---|---|
| 9 | Một năm có bao nhiêu tháng? | 1.00 | **0.00** | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh"}` |
| 10 | Nước sôi ở bao nhiêu độ C? | 1.00 | **0.00** | schema triage + `intent_confidence`, `entities` — không có ô nào chứa "100" |
| 13 | TP.HCM trước đây tên gì? | 1.00 | **0.00** | `{"intent": "hoi_thong_tin", "intent_text": "Người dùng đang hỏi về tên cũ của thành phố Hồ Chí Minh", ...}` |
| 12 | Ý nghĩa 'Có công mài sắt…' | 0.67 | 0.33 | `{"intent": "tinh_thanh", "lesson": "...kiên trì...", "example": "..."}` |
| 11 | Kể tên một trái cây nhiệt đới | 0.20 | 0.00 | `{"intent": "hoi_thong_tin", "product": "dâu tây", ...}` |

Câu 13 là bằng chứng quyết định: mô hình viết ra rằng *"Người dùng đang hỏi về tên cũ của
thành phố Hồ Chí Minh"*. Nó **hiểu** câu hỏi. Nó chỉ không có ô nào để viết "Sài Gòn" vào,
vì schema nó học được có đúng 4 khoá và không khoá nào dành cho một câu trả lời.

Đối chiếu với 10 câu **không** tụt điểm thì cơ chế lộ ra hoàn toàn. Ở những câu đó, mô hình
**tự chế ra schema mới có ô chứa đáp án**, và trả lời đúng:

* `Thủ đô của Việt Nam?` → `{"intent": "trivia", "confidence": 0.95, "answer": "Hà Nội", ...}` — 1.00
* `Ai là tác giả Truyện Kiều?` → `{"intent": "hoi_thong_tin", "answer": "...nhà thơ Nguyễn Du"}` — 1.00
* `Dịch 'Tôi thích đọc sách'` → `"I like reading books."` — 1.00
* `2 mũ 10 bằng bao nhiêu?` → `{"intent": "math", "answer": "1024"}` — **1.00, trong khi base SAI (0.00)**

Câu cuối đáng dừng lại: trên câu hỏi đó bản fine-tune **giỏi hơn** base. Đó là điều không thể
xảy ra nếu 30 step huấn luyện đã xoá mất kiến thức.

Vậy chẩn đoán đúng không phải "mô hình quên", mà là: huấn luyện dạy nó rằng **mọi đầu vào
đều là JSON có cấu trúc**, và nó khái quát hoá *cái vỏ* đó ra ngoài miền. Khi nó chế được
schema có ô `answer`, kiến thức đi qua nguyên vẹn. Khi nó rơi về đúng schema 4 khoá của tập
huấn luyện, câu trả lời không còn chỗ nào để tồn tại và điểm về 0. Nhóm regression vì thế
đang đo **"nó có chịu trả lời không"**, chứ không đo "nó có biết không".

Điều này *củng cố* đơn thuốc của deck §14.3 chứ không bác bỏ: trộn 1–5% dữ liệu phổ thông
dạy mô hình rằng tồn tại những đầu vào **không** phải ticket, tức là tấn công đúng vào cơ chế
trên. Nhưng nó cũng chỉ ra một biện pháp rẻ hơn đáng thử trước: giữ nguyên hướng dẫn tác vụ
trong prompt lúc train, để mô hình có tín hiệu rằng đây là *một* tác vụ chứ không phải *toàn
bộ thế giới*. `SIMULATION-FINDINGS.md` đã ghi lại đúng cặp số ủng hộ điều này ở tier 0.8B:
cùng adapter đó, đo với hướng dẫn còn trong prompt cho regression 0.522 thay vì 0.067.

**Một lưu ý về chính thang đo.** Nhóm regression chấm bằng keyword recall trên văn bản đầu
ra, nên nó không phân biệt được "không biết" với "biết nhưng không nói". Nó cũng không phải
thang đo nghiêm: câu `Sông nào dài nhất Việt Nam?` **cả hai** model đều trả lời "sông Hồng"
và cùng bị 0.00, còn ở câu trái cây nhiệt đới thì base trả lời "dâu tây" rồi tự đính chính
giữa chừng. Bar 0.7578 của base là điểm của một mô hình nói năng lan man được chấm nới tay,
không phải một năng lực đáng bảo vệ bằng mọi giá. Điều đó **không** cứu được bản fine-tune —
0.5889 vẫn là tụt thật và vẫn vượt dung sai 8,4 lần — nhưng nó có nghĩa là con số 0.169 nên
đọc như một *cảnh báo có phương hướng*, không phải một phép đo chính xác lượng năng lực đã mất.

`valid_trace_rate = 0.00` là kết quả **đúng như dự đoán**, không phải lỗi. Cả 250 câu trả lời
huấn luyện đều là JSON trần, không có trace nào để học; và như mục 2 cho thấy, khối
`<think></think>` rỗng đã được đóng ngay trong generation prompt. Không có gì dạy mô hình
sinh suy luận thì nó không sinh suy luận. Con số này chỉ có ý nghĩa khi làm thử thách B3 với
dữ liệu có trace thật.

**Có nên deploy không? Không, chưa.** Nhưng "không" ở đây là một chẩn đoán có địa chỉ, không
phải một thất bại: bài toán đã được giải (0.970), thứ hỏng là tác dụng phụ, và deck §14.3 nêu
sẵn thuốc — trộn 1–5% dữ liệu phổ thông vào tập huấn luyện. Nếu tôi chỉ đo nhóm target thì
tôi đã ship một mô hình phân loại ticket xuất sắc mà hễ ai hỏi nó bất cứ điều gì khác là nó
trả về JSON.

---

## 6. Định tính — và một kết quả rubric không lường trước

Trước khi liệt kê ví dụ, phải nói thẳng một điều: **trên nhóm target, fine-tune không thua
mẫu nào.** Đối chiếu từng mẫu giữa (b) và (c) trên đủ 50 mẫu:

| | số mẫu |
|---|---|
| fine-tune thắng (b) | **34** |
| hoà | 16 |
| fine-tune thua (b) | **0** |

Rubric 3.4 yêu cầu ≥2 ca fine-tune thua, và lý do của yêu cầu đó là chống cherry-pick. Trên
nhóm target tôi không thể cung cấp ca thua vì **chúng không tồn tại** — nên thay vì chọn ví
dụ, tôi trình bày toàn bộ phân bố ở trên; đó là bằng chứng mạnh hơn một cặp ví dụ, vì nó
không cho phép chọn lọc. Dữ liệu đầy đủ nằm ở `results/qualitative.json` (50 dòng) và
`results/baseline_preds.json`.

Ca thua có thật, và chúng nằm ở nhóm **regression** — 5 câu, liệt kê đầy đủ ở mục 5.1 với
nguyên văn câu trả lời. Ba trong số đó là thua trắng 1.00 → 0.00. Chúng được đưa vào đây
đúng như rubric đòi hỏi; chỉ khác là chúng không nằm ở nhóm mà rubric mặc định sẽ tìm thấy,
và điều đó bản thân nó là kết quả: bản fine-tune này không đánh đổi trong tác vụ, nó đánh đổi
**ra ngoài** tác vụ.

| # | Ticket (rút gọn) | Nhãn đúng (trường sai) | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 (i=6) | "…đặt balo laptop… **Đổi size**. Hỏi cho biết" | `doi_tra` / `thap` | `hoan_tien` / `cao` — 0.50 | **đúng cả 4 trường** — 1.00 | ✅ FT thắng +0.50 |
| 2 (i=19) | "…nồi chiên không dầu… **Vỡ khi nhận**" | `san_pham_loi` / `cao` | `hoan_tien` — 0.50 | **đúng cả 4 trường** — 1.00 | ✅ FT thắng +0.50 |
| 3 (i=36) | "…tai nghe bluetooth… **Giá bao nhiêu**" | `hoi_thong_tin` / `thap` | `doi_tra` / `trung_binh` — 0.50 | `hoi_thong_tin` / `trung_binh` — 0.75 | ✅ FT thắng +0.25, vẫn sai `urgency` |
| 4 (i=3) | "…bình giữ nhiệt… Chưa thấy tiền" | `hoan_tien` / **`thap`** | `trung_binh` — 0.75 | `trung_binh` — 0.75 | ❌ **FT sai**, hoà với (b) |
| 5 (i=12) | "…áo khoác gió… Bị lỗi. **Khi nào tiện**" | `san_pham_loi` / **`thap`** | `trung_binh` — 0.75 | `trung_binh` — 0.75 | ❌ **FT sai**, hoà với (b) |
| 6 (reg. 9) | "Một năm có bao nhiêu tháng?" | `12` | trả lời "12 tháng" — 1.00 | `{"intent": "hoi_thong_tin", …}` — 0.00 | ❌ **FT THUA −1.00** |
| 7 (reg. 13) | "TP.HCM trước đây có tên là gì?" | `Sài Gòn` | trả lời "Sài Gòn" — 1.00 | phân loại câu hỏi, không trả lời — 0.00 | ❌ **FT THUA −1.00** |

**Có mẫu chung ở các ca fine-tune sai không? Có, và nó tuyệt đối.** Cả **6/6** mẫu mà
fine-tune không đạt điểm tuyệt đối đều sai **đúng một trường** — `urgency` — và **đúng một
hướng**: nhãn `thap` bị đoán thành `trung_binh`. Không mẫu nào sai `intent`, không mẫu nào
sai `product`, không mẫu nào sai `sentiment`, không mẫu nào đoán nhầm theo hướng ngược lại.

Đây không phải nhiễu, đây là **mode collapse về lớp đa số**. `trung_binh` là nhãn urgency phổ
biến nhất trong tập huấn luyện, và với các ticket mang tín hiệu "không gấp" tinh tế —
"Khi nào tiện", "Hỏi cho biết", "Chưa thấy tiền" — mô hình chọn giá trị an toàn thay vì đọc
tín hiệu. Đáng chú ý là baseline (b) sai **y hệt** ở cả ba mẫu đó, nên chúng thành hoà chứ
không thành thua. Nghĩa là fine-tune không hề tạo ra lỗi mới; nó chỉ **không sửa được** lỗi
mà prompt engineering cũng không sửa được.

Điều đó chỉ thẳng vào việc cần làm tiếp, và nó không phải chuyện của mô hình: cần thêm mẫu
huấn luyện `urgency=thap` với các cách diễn đạt "không gấp" đa dạng hơn. Đây là lỗi dữ liệu,
không phải lỗi cấu hình — và không có bảng định tính này thì con số 0.970 sẽ không nói cho
tôi biết điều đó.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không deploy** bản fine-tune này, dù nó thắng baseline mạnh nhất của
chính nó tới +0.205 trên nhóm target — vì nó đánh mất 0.169 năng lực tổng quát, vượt dung sai
8,4 lần. Đó là một quyết định khó chịu đúng theo cách lab thiết kế: nếu tôi chỉ đo cái tôi
đang tối ưu, tôi đã có một chiến thắng sạch sẽ và một mô hình hỏng. Cổng bốn nhóm tồn tại
chính xác để ngăn khoảnh khắc đó.

Nhưng "đánh mất năng lực tổng quát" là cách nói thiếu chính xác, và mục 5.1 cho thấy vì sao.
Kiến thức không mất: cùng bản fine-tune ấy trả lời đúng "Hà Nội", "Nguyễn Du", và thậm chí
trả lời đúng `2^10 = 1024` ở câu mà base **tính sai**. Thứ bị bắt cóc là *định dạng đầu ra* —
huấn luyện dạy nó rằng mọi đầu vào đều là JSON có cấu trúc, nên khi nó tự chế được schema có
ô `answer` thì kiến thức đi qua nguyên vẹn, còn khi nó rơi về schema triage 4 khoá thì câu
trả lời không còn chỗ nào để tồn tại. Cả 5 câu tụt điểm đều thuộc trường hợp thứ hai. Đây là
một chẩn đoán khác hẳn "quên", và nó dẫn tới cách sửa khác: không phải dạy lại kiến thức, mà
dạy lại rằng tồn tại những đầu vào **không** phải ticket.

Đâu là đòn bẩy thật? Xếp theo mức độ ảnh hưởng tôi đo được: **learning rate** đứng đầu và bỏ
xa phần còn lại — sai 10 lần là mất trắng, `target 0.000` và `format 0.000`, một tham số đưa
run từ tốt nhất xuống không dùng được. Kế đến là **chất lượng và thành phần dữ liệu**, thể
hiện ở cả hai đầu: dữ liệu không có replay gây ra toàn bộ lý do FAIL, và dữ liệu thiếu mẫu
`urgency=thap` gây ra toàn bộ 6 lỗi còn lại. **Mask** đứng thứ ba — nó là điều kiện cần, mask
sai thì mọi con số sau đều vô nghĩa, nhưng khi đã đúng thì nó không còn là biến số nữa.
**Vị trí adapter** đứng cuối, và tôi phải trung thực rằng lab này **không** chứng minh được
nó là đòn bẩy: ở ngân sách tham số khớp nhau, `q,v @ r=283` hoà với `text-linear @ r=16` ở
0.970. Deck §10.2 dự đoán `correct` thắng; số của tôi cho hoà. Nhiều khả năng do đụng trần
tác vụ, nhưng "nhiều khả năng" không phải bằng chứng, và tôi ghi lại kết quả chứ không ghi
lại điều tôi mong đợi.

Điều làm tôi đổi cách nghĩ nhiều nhất không phải một con số mà là một sự bất đồng: `attn_only`
có train loss **thấp hơn** `correct` (0.5378 so với 0.6267) trong khi trên thang đo tác vụ
hai bên hoà. Chấm bằng loss thì tôi đã xếp hạng cấu hình mà deck gọi là Lỗi #1 lên đầu bảng.
Chỉ số thay thế không sai một cách ngẫu nhiên — nó sai một cách có hệ thống, theo hướng
thưởng cho việc học thuộc bề mặt.

**Ba điều tôi học được:**

1. **Chỉ số thay thế có thể xếp hạng ngược, không chỉ xếp hạng kém.** Tôi từng nghĩ train
   loss là tín hiệu thô nhưng cùng chiều với chất lượng. Ở đây nó ưu ái cấu hình rank cao gắn
   ở ít vị trí — cấu hình khớp bề mặt tốt hơn — trong khi thang đo tác vụ cho hoà. Từ nay
   mọi so sánh cấu hình tôi làm sẽ chấm bằng thang đo cuối, và loss chỉ dùng để phát hiện
   lỗi thô như `wrong_lr`.

2. **Lỗi chỉ tồn tại trên phần cứng mặc định là lỗi khó thấy nhất.** Trong lúc chạy lab tôi
   gặp hai bug NB6 mà mọi lần kiểm chứng trước đó đều bỏ sót: `merge_and_unload()` trả về
   chính module base chứ không phải bản sao, nên `del merged` không giải phóng gì và lần nạp
   base thứ hai bị đẩy sang CPU; và `merge_check.json` — artefact duy nhất được chấm — bị ghi
   *sau* một lần lưu 8,4 GB không ai đọc, khiến một phán quyết đã tính xong bị mất khi người
   chạy tưởng máy treo. Cả hai vô hình trên model 0.8B, nơi hai bản model cùng vừa VRAM. Lần
   verify duy nhất trước đó dùng đúng model nhỏ ấy.

3. **Một cổng liêm chính báo sai còn tệ hơn không có cổng.** `make verify` trên bản clone
   Windows sạch buộc tội tôi sửa tập eval — cả bốn file. Thực tế là `core.autocrlf=true` đổi
   LF thành CRLF khi checkout; chuẩn hoá ngược lại thì cả bốn hash khớp chính xác. Phản ứng
   tự nhiên khi thấy màu đỏ đó là sinh lại `checksums.json` cho hết đỏ — tức là làm đúng cái
   hành vi mà cổng sinh ra để chặn. Từ đó tôi rút ra: kiểm tra tính toàn vẹn phải so *nội
   dung*, và quy ước xuống dòng không phải nội dung.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% dữ liệu phổ thông vào tập huấn luyện rồi
train lại đúng cấu hình `correct` với cùng 30 step, để xem regression có quay về trong dung
sai 0.020 mà vẫn giữ target quanh 0.97 hay không — đó là kiểm chứng trực tiếp cho đơn thuốc
của deck §14.3, và là thứ duy nhất chắn giữa lần chạy này và một quyết định deploy. Việc thứ
hai: bổ sung mẫu `urgency=thap` để tấn công mode collapse ở mục 6. Việc thứ ba: lặp lại đối
chứng vị trí-vs-rank trên một tác vụ chưa đụng trần, vì ở 0.970 tôi không tách được hai cấu
hình.

---

## Phụ lục — thưởng đã làm

- [x] **B1 — NB6 merge: làm được một nửa.** `results/merge_check.json`:
      `before_merge 0.97 → after_merge 0.97, delta 0.0` trên **50 mẫu** (không phải 20).
      Merge chính xác về số học, không tụt điểm. Phần **hot-swap không chạy được**: nạp base
      lần thứ hai trong cùng tiến trình bị `accelerate` đẩy 13 lớp cuối sang CPU và PEFT từ
      chối gắn adapter (`ValueError: We need an offload_dir`). Nguyên nhân đã tìm ra và vá
      trong repo, nhưng lần chạy có điểm này diễn ra trước bản vá nên tôi báo cáo đúng những
      gì đã đo được, không nhiều hơn.
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse — không làm được với corpus mặc định: cả 250 câu trả lời là
      JSON trần, nên `masked-think` và `response-only` cho ra mask giống hệt `assistant-only`
      (xem ghi chú trong `.env.example`). Cần dữ liệu có trace thật.
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub

### Ghi chú trung thực về lần chạy

* **Ba unit test đỏ trên Colab.** `verify.py --smoke` báo `3 failed, 115 passed`. Đúng ba
  test bị skip ở máy không có torch — và cả ba đều pass ở local sau khi cài torch 2.13.0+cpu.
  Chúng canh đường precision fp16 (F-15, F-23), tức chính đường lần chạy này train trên đó.
  NB3 và NB4 đều hoàn tất bình thường nên chúng không tiên đoán hỏng hóc thật, nhưng tôi
  không có lời giải thích cho việc chúng đỏ trên T4 và xanh ở nơi khác, nên ghi lại ở đây
  thay vì lờ đi.
* **Sửa trong repo trong lúc làm lab.** NB2 không lưu dự đoán từng mẫu của baseline (b) nên
  mục 6 không thể lập bảng — đã thêm `results/baseline_preds.json`. NB5 xếp hạng ví dụ định
  tính theo điểm tuyệt đối của fine-tune thay vì theo `delta` so với (b) — hai câu hỏi khác
  nhau, đã sửa. Hai bug NB6 và lỗi CRLF nêu ở mục 7. Tất cả đều có commit riêng kèm nguyên
  nhân, và ba lỗi trong số đó có test chặn hồi quy.
* **Dự đoán từng mẫu của nhóm regression** không được NB2 hay NB5 lưu lại — cả hai chỉ giữ
  giá trị trung bình, nên nhóm quyết định phán quyết FAIL lại là nhóm duy nhất không kiểm
  chứng được. Đã bổ sung `scripts/dump_regression_preds.py` và chạy sau khi NB5 hoàn tất; nó
  dựng lại đúng lời gọi của NB5 (`system=None`, 96 token) nên hai giá trị trung bình khớp
  chính xác với `baselines_frozen.json` và `verdict.json` (0.7578 và 0.5889). Kết quả là mục
  5.1 — và nó đổi kết luận từ "thảm hoạ quên" sang "định dạng đầu ra bị bắt cóc", một chẩn
  đoán khác hẳn. Nó chạy **sau** khi mọi thứ đã đóng băng và chỉ đọc, nên không đụng vào phép
  so sánh được chấm.
