# Nhật ký quyết định

## Đã chốt

| Ngày | Quyết định | Lý do |
|---|---|---|
| 25/10/2026 | Chọn paper "When Not to Trust Language Models" (ACL 2023) | Cô đã duyệt; có code công khai; chạy được mô hình gốc trên GPU miễn phí |
| 25/10/2026 | Hướng mở rộng chính: mô hình đời mới (Qwen2.5 bản base) | Đúng ví dụ của cô; chi phí vừa phải |
| 26/10/2026 | Code gốc chạy bằng `--sample 300` trên file `popQA.tsv` đầy đủ, không lọc file xuống 300 dòng. Notebook `upstream_kaggle.ipynb` tự kiểm id kết quả khớp `data/subset_300_ids.txt` rồi mới coi là đúng | Lọc file khiến 15 ví dụ mẫu chỉ bốc trong 300 dòng (loại `color` chỉ có 2 dòng), khác paper và khác notebook chính. Với seed 633, code gốc chọn đúng 300 id đó (đã thử trên CPU) |
| 05/11/2026 | #12: 15 ví dụ mẫu của BM25 **kèm đoạn văn** top-1 của chính ví dụ đó. Khi chạy code gốc, bản chép của `run_model.py` tra đoạn văn cho ví dụ mẫu bằng nguyên văn câu hỏi, như câu chính (không sửa `upstream/`) | Code gốc có sẵn nhánh kèm đoạn văn cho ví dụ mẫu, nhưng với file BM25 hiện có thì khóa tra không khớp (viết thường, bỏ `?`) nên 0/14.267 ví dụ có đoạn (`upstream/KNOWN_ISSUES.md` mục 1). Số đo trên 300 câu GPT-Neo 1.3B, code gốc fp32: không kèm 24,33% (26/10), kèm 25,67% (06/11); đổi lại 141/300 prompt dài quá 2048 token và bị cắt từ trái (`results/RUNS.md`) |
| 05/11/2026 | #13: chạy **đủ 14.267 câu**, không dùng tập con. Tập 300 câu `data/subset_300_ids.txt` chỉ để chạy thử và đối chứng | Đo được ở lượt chạy 06–07/11: GPT-Neo 1.3B BM25 kèm đoạn văn 2,63 giây/câu, đủ bộ mất 10,4 giờ, vừa 1 phiên Kaggle; 2.7B cần 2 phiên, notebook lưu từng câu và tự chạy tiếp. Đủ bộ thì adaptive (ngưỡng theo 10 bin độ nổi tiếng cho từng loại quan hệ) có đủ câu mỗi bin và so thẳng được với paper |
| 10/11/2026 | #32: hướng phụ là **C**, adaptive theo độ tự tin của mô hình (cột `conf`) thay cho độ nổi tiếng (#34). Bỏ hướng B, xếp hạng lại đoạn BM25 (#33 đóng, not planned). Tân chốt | C sửa đúng điểm yếu của paper: ngoài đời không biết trước câu hỏi nổi tiếng cỡ nào, còn độ tự tin thì mô hình nào cũng tự cho được. Không cần GPU: lượt vanilla đủ 14.267 câu của cả 9 mô hình đã có `conf`; 8 mô hình đã đủ BM25 để so ngay, Qwen2.5 3B so khi xong BM25. B phải chạy lại mô hình trên GPU. C nối được với hướng chính: mô hình đời mới có biết mình không biết tốt hơn không |

## Đang chờ

| Issue | Câu hỏi | Chốt dựa vào |
|---|---|---|
| #5 | Cô có chấp nhận phạm vi tái lập (mô hình nhỏ, không GPT-3, không Contriever) không? | Câu trả lời của cô |
