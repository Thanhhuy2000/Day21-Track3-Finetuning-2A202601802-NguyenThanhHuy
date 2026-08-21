# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Fine-tune thắng baseline (b) tới +0.205 mà vẫn **trượt**. Tôi vào lab với giả định ngầm rằng
"thắng bài toán" và "đạt" là một; hoá ra chúng là hai câu hỏi khác nhau, và cái thứ hai mới
quyết định có deploy được hay không. Ngạc nhiên thứ hai, kỹ thuật hơn: `attn_only` có train
loss **thấp hơn** `correct` (0.5378 so với 0.6267) trong khi trên thang đo tác vụ hai bên hoà
ở 0.970. Nếu xếp hạng bằng loss, tôi đã kết luận rằng cấu hình mà deck gọi là Lỗi #1 là cấu
hình tốt nhất. Chỉ số thay thế không lệch ngẫu nhiên — nó lệch có hệ thống, về phía thưởng
cho việc khớp bề mặt.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không đúng dự đoán chút nào. Tôi tưởng NB4 sẽ là chỗ tốn thời gian nhất — nó đúng là dài nhất
(45–60 phút) nhưng chạy một mạch không sự cố. Thời gian thật sự mất là ở **NB6**, notebook
*tuỳ chọn* đáng lẽ mất 10 phút: một lần ghi 8,4 GB trông y hệt máy treo vì tqdm hiện
`0% 0/1` bất động suốt 13 phút (chỉ có một shard nên thanh tiến trình không có gì để nhích),
rồi sau khi vá xong lại gặp tiếp lỗi VRAM ở phần hot-swap. Ba điểm thưởng ngốn nhiều thời
gian hơn cả bốn run huấn luyện cộng lại.

Bài học rút ra không phải "NB6 tệ" mà là: chi phí thật của một bước nằm ở **chỗ nó hỏng theo
kiểu không giống hỏng**. Một traceback đọc mất 30 giây; một thanh tiến trình đứng yên ngốn 13
phút chỉ để quyết định xem có nên đợi tiếp hay không.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Rằng so sánh "trước và sau fine-tune" là một phép so sánh có nghĩa. Baseline (a) — base model
với prompt ngây thơ — đạt `target 0.000` và `format 0.000`. So với nó thì bản fine-tune của
tôi thắng vô hạn, và con số đó **không nói lên điều gì cả**, vì (a) thậm chí không trả về
JSON. Mốc đúng là (b): cùng model đó, prompt viết tử tế, 0.765 với JSON hợp lệ 100% và
**nhanh hơn (a) 3,2 lần**. Khoảng cách giữa "thắng (a)" và "thắng (b)" là toàn bộ khác biệt
giữa một con số để khoe và một quyết định kỹ thuật.

Điều thứ hai tôi không còn tin: rằng fine-tune "thêm" kỹ năng. Nó **dịch chuyển** mô hình,
và tôi còn hiểu sai cả cách nó dịch chuyển. Thấy regression tụt 0.758 → 0.589, tôi kết luận
ngay là thảm hoạ quên. Đến khi đọc từng câu trả lời thì hoá ra không phải: cùng bản fine-tune
ấy vẫn trả lời đúng "Hà Nội", "Nguyễn Du", và trả lời đúng `2^10 = 1024` ở câu mà base
**tính sai**. Kiến thức còn nguyên. Thứ nó đánh mất là *chỗ để đặt câu trả lời* — nó học được
rằng mọi đầu vào đều là JSON 4 khoá, nên với câu "TP.HCM trước đây tên gì?" nó viết ra
`intent_text: "Người dùng đang hỏi về tên cũ của thành phố Hồ Chí Minh"` rồi dừng. Nó hiểu
câu hỏi và không có ô nào để viết "Sài Gòn".

Bài học thật nằm ở chỗ tôi suýt không học được nó: một con số tổng hợp cho tôi *một* câu
chuyện hợp lý, và tôi đã định dừng ở đó. Chỉ khi nhìn 15 câu trả lời cụ thể thì câu chuyện
mới đổi — và cách sửa cũng đổi theo.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng để dựng môi trường, đọc hiểu cấu trúc repo, chẩn đoán lỗi lúc chạy, và viết report từ
`results/`. Chỗ nó sai, theo thứ tự nghiêm trọng giảm dần:

* **Khẳng định sai về Colab.** Nó nhiều lần bảo tôi "chạy ô mới, không ảnh hưởng ô đang chạy".
  Colab chỉ có một kernel: ô mới **xếp hàng** chứ không chạy song song. Vì thế một lệnh chẩn
  đoán tôi bấm chạy đã nằm chờ suốt mà không ai biết. Nó tự phát hiện và sửa khi tôi hỏi cách
  chạy lệnh.
* **Chẩn đoán vội "NB6 treo".** Thực tế nó đang ghi bình thường, chỉ chậm — file tạm 8,41 GB
  chứng minh điều đó. Suýt nữa thì ngắt oan một tiến trình khoẻ mạnh.
* **`git add -A` quét luôn file zip 124 MB** vào commit, và GitHub từ chối nhận. Phải reset
  lại hai commit và thêm `*.zip` vào `.gitignore`.

Chỗ nó có ích nhất lại không phải viết code, mà là **không chịu bỏ qua một cảnh báo**: ba
unit test đỏ, cột baseline (b) bị vứt đi, checksum lệch. Cái thứ hai nếu bỏ qua thì mục 6 của
report đã không lập được bảng; cái thứ ba nếu "sửa" theo cách dễ nhất — sinh lại
`checksums.json` — thì chính là hành vi gian lận mà cổng ấy sinh ra để chặn.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Viết prompt tốt nhất tôi có thể viết cho base model, và đo nó trên một tập eval đã đóng băng
— **trước khi** đụng tới GPU. Đó là baseline (b), và nó trả lời câu hỏi đắt nhất trước tiên:
*có cần fine-tune không?* Ở lab này (b) đạt 0.765 với JSON hợp lệ 100% và latency chỉ bằng
1/3 của (a). Nếu ngưỡng nghiệm thu của khách hàng là 0.75 thì dự án đã xong ngay tại đó, với
chi phí bằng không và không có món nợ vận hành nào.

Bước thứ hai, cũng làm trước khi train: định nghĩa **cái không được phép hỏng**. Ở đây là 15
câu hỏi phổ thông, và đúng chúng — chứ không phải nhóm target — là thứ bác bỏ mô hình của
tôi. Không có nhóm đó, tôi đã giao cho khách một bộ phân loại ticket 0.970 mà hễ hỏi bất cứ
điều gì khác là nó trả về JSON triage.
