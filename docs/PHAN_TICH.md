# Phân tích (10%)

Mục 1 và 2 làm ở issue #35 và #36. File này viết tiếp khi các issue đó xong.

## 3. Những lần retrieval gây hại (issue #37)

### Câu hỏi

Kết luận 3 của paper nói retrieval giúp câu ít nổi tiếng nhưng có thể hại câu nổi tiếng. Câu hỏi ở đây: những câu vanilla trả lời đúng mà BM25 làm sai ("bị hại") là câu nào, và đoạn văn nào đánh lừa mô hình?

### Tóm tắt

- Bị hại: 2,3% đến 4,5% số câu (bảng 2×2 ở `docs/BANG_TAI_LAP.md`). **35,6% đến 54,8% số câu bị hại nằm ở một phần năm câu nổi tiếng nhất**, trong khi nếu đều thì chỉ 20%.
- Nguyên nhân gần nhất: đoạn văn BM25 **không nói về chủ thể của câu hỏi**. Đoạn đó hoặc nói về một thực thể khác có tên giống (cả 9 mô hình cùng chép theo), hoặc là một đoạn "dùng chung" chọn cho hàng trăm chủ thể khác nhau.
- **Một phần ba câu hỏi (4.787 trên 14.267, tức 33,6%) nhận một đoạn dùng chung**, với recall@1 chỉ 1,0% (so với 38,0% ở các câu còn lại). Ở các câu này BM25 làm độ chính xác giảm ở cả 9 mô hình; ở phần còn lại BM25 tăng 13,8 đến 16,6 điểm.

Một câu cho slide: *retrieval không hại ngẫu nhiên; nó hại khi đoạn văn nói về một thực thể khác trùng tên, hoặc khi BM25 trả về một đoạn chung chung cho cả nghìn câu hỏi.*

### Thống kê 1: ai bị hại, theo độ nổi tiếng

Chia mọi câu thành 5 nhóm bằng số câu theo log(`s_pop`), gộp mọi loại quan hệ (nhóm 1 ít nổi tiếng nhất, giống `docs/BANG_MO_RONG.md`). Mỗi ô là tỉ lệ câu của nhóm bị hại.

| Mô hình | Nhóm 1 | Nhóm 2 | Nhóm 3 | Nhóm 4 | Nhóm 5 | Phần câu bị hại nằm ở nhóm 5 |
|---|---|---|---|---|---|---|
| GPT-Neo 1.3B | 2,4% | 1,4% | 1,7% | 1,4% | 4,6% | 40,1% |
| GPT-Neo 2.7B | 2,1% | 1,6% | 2,0% | 1,9% | 6,7% | 47,2% |
| OPT 1.3B | 2,6% | 1,9% | 1,6% | 1,9% | 5,6% | 41,2% |
| OPT 2.7B | 2,2% | 1,8% | 1,7% | 2,1% | 7,7% | 49,5% |
| OPT 6.7B (fp16) | 2,1% | 1,9% | 2,5% | 3,1% | 10,5% | 52,3% |
| GPT-J 6B (fp16) | 2,4% | 1,6% | 2,5% | 3,2% | 11,8% | 54,8% |
| Qwen2.5 0.5B | 2,5% | 1,7% | 2,1% | 1,7% | 4,4% | 35,6% |
| Qwen2.5 1.5B | 2,9% | 2,6% | 2,8% | 3,3% | 8,3% | 41,3% |
| Qwen2.5 3B | 2,4% | 2,0% | 2,3% | 4,7% | 11,0% | 49,1% |

- Bốn nhóm đầu bị hại 1,4% đến 4,7%; nhóm nổi tiếng nhất 4,4% đến 11,8%, cao nhất ở GPT-J, Qwen2.5 3B và OPT 6.7B (10,5% đến 11,8%).
- Nhóm nổi tiếng nhất bị hại nhiều không phải vì retrieval tệ hơn ở đó, mà vì **ở đó vanilla trả lời đúng nhiều nhất, nên có nhiều thứ để mất**: với GPT-J, 46,2% mọi câu vanilla đúng nằm ở nhóm 5. Tỉ lệ câu vanilla đúng bị làm sai (`lost_share`) ở nhóm 5 là 20,5% đến 31,7%, không cao hơn nhóm 3 và 4 (25,2% đến 36,4%); thấp nhất là nhóm 1 (12,5% đến 16,1%).

### Thống kê 2: vì sao, nhìn từ đoạn văn

| Mô hình | Số câu bị hại | recall@1 của ô hại | Câu trả lời BM25 nằm nguyên văn trong đoạn: ô hại | ô giúp |
|---|---|---|---|---|
| GPT-Neo 1.3B | 327 | 0,30 | 67% | 90% |
| GPT-Neo 2.7B | 407 | 0,25 | 66% | 90% |
| OPT 1.3B | 391 | 0,34 | 63% | 90% |
| OPT 2.7B | 442 | 0,23 | 66% | 90% |
| OPT 6.7B (fp16) | 574 | 0,19 | 60% | 90% |
| GPT-J 6B (fp16) | 613 | 0,15 | 64% | 90% |
| Qwen2.5 0.5B | 357 | 0,27 | 70% | 94% |
| Qwen2.5 1.5B | 571 | 0,22 | 54% | 77% |
| Qwen2.5 3B | 640 | 0,12 | 66% | 92% |

- recall@1 của ô hại là 0,12 đến 0,34 (định nghĩa và so với paper ở `docs/BANG_TAI_LAP.md`): trong 66% đến 88% câu bị hại, đoạn văn **không chứa đáp án đúng**.
- Cột cuối cho biết câu trả lời sai của BM25 có nằm nguyên văn trong đoạn không (cận dưới của "mô hình chép từ đoạn", vì chỉ đếm khớp chuỗi): 54% đến 70% ở ô hại, so với 77% đến 94% ở ô giúp. Nghĩa là phần lớn câu sai là mô hình **chép một tên từ đoạn văn sai**, không phải tự bịa.

### Thống kê 3: đoạn dùng chung (phát hiện chính)

Chỉ có 8.051 đoạn khác nhau được chọn làm top-1 cho 14.267 câu hỏi. Một đoạn Wikipedia về một thực thể có thể hợp với nhiều loại quan hệ của chính thực thể đó, nhưng không thể hợp với câu hỏi về nhiều chủ thể khác nhau. Ta gọi **đoạn dùng chung** là đoạn được chọn làm top-1 cho câu hỏi về ít nhất 5 chủ thể (`subj`) khác nhau. Có 64 đoạn như vậy, phủ 4.787 câu (33,6%). Ngưỡng 3 hay 10 chủ thể cho 31% đến 36% số câu, nên kết quả không phụ thuộc vào ngưỡng.

| Mô hình | Câu có đoạn dùng chung: vanilla → BM25 | Các câu còn lại: vanilla → BM25 | Phần câu bị hại thuộc đoạn dùng chung |
|---|---|---|---|
| GPT-Neo 1.3B | 1,4% → 0,8% | 16,4% → 32,1% | 15,0% |
| GPT-Neo 2.7B | 1,6% → 1,4% | 17,8% → 33,5% | 12,8% |
| OPT 1.3B | 2,0% → 1,7% | 17,5% → 32,2% | 12,8% |
| OPT 2.7B | 2,9% → 1,8% | 17,4% → 34,1% | 18,6% |
| OPT 6.7B (fp16) | 5,1% → 3,0% | 21,0% → 35,6% | 25,1% |
| GPT-J 6B (fp16) | 4,9% → 1,9% | 21,7% → 35,8% | 28,1% |
| Qwen2.5 0.5B | 1,4% → 1,1% | 17,4% → 33,4% | 10,6% |
| Qwen2.5 1.5B | 3,1% → 1,8% | 22,4% → 36,6% | 16,6% |
| Qwen2.5 3B | 5,6% → 2,4% | 24,8% → 38,5% | 27,2% |

- **Câu có đoạn dùng chung** (4.787 câu): recall@1 là 1,0%; BM25 làm độ chính xác giảm ở cả 9 mô hình (vanilla 1,4% đến 5,6%, còn 0,8% đến 3,0%).
- **Các câu còn lại** (9.480 câu): recall@1 là 38,0% và BM25 tăng độ chính xác 13,8 đến 16,6 điểm. Vậy retrieval giúp rất nhiều ở hai phần ba câu hỏi, và hại ở phần ba còn lại.
- Đoạn dùng chung chiếm 10,6% đến 28,1% số câu bị hại, nhiều nhất ở GPT-J (28,1%), Qwen2.5 3B (27,2%) và OPT 6.7B (25,1%). Đây cũng là ba mô hình trả lời đúng nhiều nhất các câu này khi không có đoạn văn (vanilla 4,9% đến 5,6%), nên có nhiều thứ để mất.
- Mỗi đoạn dùng chung phục vụ đúng **một loại quan hệ** và chứa đúng từ của loại đó. Năm đoạn dùng nhiều nhất: đoạn về Sophie Fante ("a television producer") cho 1.156 câu producer, 984 chủ thể khác nhau; Reed De Rouen ("actor and screenwriter") cho 952 câu screenwriter; "The Winning Side" (tiểu thuyết) cho 644 câu author; "What a Life!" (album) cho 528 câu genre; "Terror Nation" (phim tài liệu Doctor Who) cho 517 câu director. Với các câu có đoạn dùng chung, đoạn đó chứa từ của loại quan hệ ở 98% câu producer, 99% author, 94% screenwriter, 94% genre, 91% director, 86% composer, 100% religion. Gợi ý rằng khi BM25 không khớp được tên chủ thể thì nó chọn đoạn khớp từ khóa của loại quan hệ. Chưa kiểm vì sao tên chủ thể không khớp (ví dụ trang về chủ thể đó có thể không nằm trong kho đoạn văn).

| Loại quan hệ | Số câu | Câu có đoạn dùng chung | Tỉ lệ |
|---|---|---|---|
| producer | 1520 | 1196 | 79% |
| screenwriter | 1999 | 1039 | 52% |
| author | 1514 | 764 | 50% |
| composer | 978 | 348 | 36% |
| genre | 1619 | 572 | 35% |
| director | 1999 | 695 | 35% |
| religion | 338 | 95 | 28% |
| mother | 187 | 13 | 7% |
| occupation | 532 | 24 | 5% |
| father | 570 | 23 | 4% |
| sport | 547 | 9 | 2% |
| country | 838 | 9 | 1% |
| capital of, capital, color, place of birth | 1626 | 0 | 0% |

### Ba ví dụ

Cả ba câu bị hại ở **cả 9 mô hình**: vanilla đúng, BM25 sai. Đoạn văn trích từ Wikipedia (CC BY-SA).

**Ví dụ 1: "What is the capital of Samoa?"** (loại `capital`, `s_pop` 79.062, nhóm nổi tiếng thứ 5). Đáp án: Apia. Cả 9 mô hình vanilla trả lời Apia; có BM25 thì cả 9 trả lời **Pago Pago**.
> American Samoa Community College is a two-year WASC college located in the village of Mapusaga, on the island of Tutuila in American Samoa [...]. Pago Pago, the capital of the territory and its center of commerce, is approximately 4.75 miles to the northeast of the school.

Đoạn văn nói về một trường đại học ở American Samoa, không phải về nước Samoa. BM25 khớp từ "Samoa" và "capital". Đoạn có hẳn một câu "Pago Pago, the capital of the territory", trông như câu trả lời cho câu hỏi. Mô hình không kiểm "the territory" có phải là Samoa không mà chép luôn. **Thực thể có tên chứa tên chủ thể.**

**Ví dụ 2: "What sport does Albert Cox play?"** (loại `sport`, `s_pop` 71, nhóm 1). Đáp án: association football. Cả 9 mô hình vanilla trả lời football; có BM25 thì 7 mô hình trả lời **ice hockey**, 2 mô hình (GPT-Neo 1.3B và 2.7B) trả lời hockey.
> Albert Edward Cox (July 19, 1902 – May 10, 1985) was a Canadian professional ice hockey goaltender. He was born in London, Ontario. [...]

Có hai người tên Albert Cox; chủ thể của PopQA là cầu thủ bóng đá, đoạn văn nói về thủ môn khúc côn cầu. Chỉ cần tên thì không thể phân biệt hai người, kể cả người đọc rất giỏi. PopQA cũng có sẵn 881 văn bản câu hỏi trùng nhau. **Thực thể trùng tên.**

**Ví dụ 3: "In what city was Léon-Lévy Brunswick born?"** (loại `place of birth`, `s_pop` 122, nhóm 1). Đáp án: Paris. Cả 9 mô hình vanilla trả lời Paris (đúng); có BM25 thì cả 9 bỏ Paris.
> Frederick Dibblee (9 December 1753 – 17 May 1826) was a Canadian Church of England clergyman [...]. He was born in Stamford, Connecticut. [...] In 1784, he moved to what would become the new province of New Brunswick and finally settled in Woodstock, New Brunswick in 1788.

Đoạn văn không liên quan gì tới chủ thể; BM25 chỉ khớp chữ "Brunswick" trong "New Brunswick". Bảy mô hình chép một địa danh từ đoạn (Woodstock, Woodstock New Brunswick hoặc Stamford, Connecticut); hai mô hình GPT-Neo bịa ra địa danh Pháp không có trong đoạn (Saint-Jean-sur-Richelieu, Saint-Jean-de-Maurienne). Mô hình **biết đáp án đúng nhưng vẫn tin đoạn văn hơn**, dù đoạn không liên quan. **Trùng một từ trong tên.**

### Đọc tay: các câu bị hại nhiều nhất

**18 câu bị hại ở cả 9 mô hình** (19 dòng, vì PopQA có hai dòng cùng câu E.T. với `id` khác nhau). Không câu nào có đoạn dùng chung, và cả 18 đều có đoạn nói về một thực thể khác:
- **Trùng tên hoặc tựa (4 câu):** E.T. the Extra-Terrestrial (đoạn nói về album sách nói cùng tên, sản xuất bởi Quincy Jones), Bourbon Restoration (quán cà phê ở Madrid, "Bourbon Restoration ở Tây Ban Nha"), Albert Cox, Vladimir Yakovlev (một Vladimir Yakovlev khác, tướng Nga).
- **Chung một từ trong tên (14 câu):** Jan van de Cappelle (Jan van de Velde), Irving Amen (ban nhạc Amen), Western Hockey League (Hockey WA ở Australia), Lincoln (Robert Symonds, phó giám đốc nhà hát ở Lincoln Center), Naathan Phan (Michelle Phan), Legend (The Legend of Lyon Flare), Léon-Lévy Brunswick, Scheidt (Mike Scheidt), Pamela Winslow (Pamela Sheyne), Cora (Cora Cross), Chase (Diane Chase), Samoa, Anthony Sharp (Bree Sharp), Wright (Luther Wright).

**20 câu bị hại ngẫu nhiên của GPT-J** (seed 633, trong 613 câu), đọc tay:

| Loại | Số câu | Ví dụ |
|---|---|---|
| Đoạn dùng chung, không liên quan chủ thể | 5 | "Terror Nation" cho On the Road, Annie, The Kingdom; đoạn về tác giả Doctor Who cho The Voyage Out; "Scythian religion" cho Bede |
| Đoạn về thực thể khác có tên giống | 10 | Rand Paul (làng Rand), David Campese (Bruno Campese), Lysis (Lysis of Taras), Georgia (Savannah, bang Georgia thay cho nước Georgia), Serenity (album "Serenity is the Devil"), Footes (sân Footes Lane), Hugo Rubio (Ricky Rubio), Susan Kent, Lodhi dynasty, Washington |
| Đoạn không liên quan, không phải đoạn dùng chung | 1 | Grease (nữ diễn viên Doctor Who) |
| Đoạn nhắc tới chủ thể nhưng nói về thực thể liên quan, không có đáp án | 3 | Charles Washington (ngôi nhà Happy Retreat), Sheffield United (nhóm cổ động viên), York City F.C. (một cầu thủ) |
| Đoạn có đáp án nhưng mô hình chọn nhầm | 1 | The Bourne Legacy (đoạn có "Robert Ludlum" nhưng mô hình chọn tác giả khác của tiểu thuyết khác) |

Vậy **16 trên 20 (80%)** đoạn văn nói về chuyện khác, 3 trên 20 nói về thực thể liên quan nhưng thiếu đáp án, và chỉ 1 trên 20 thật sự chứa đáp án (bài toán của issue #38). Theo quy tắc chấm thì 3 trên 20 có đáp án trong đoạn (đúng với recall@1 của GPT-J là 0,15), nhưng 2 trong 3 câu đó là dương tính giả của so khớp chuỗi con ("red" nằm trong "retired" ở York City F.C.; "US" nằm trong "multi-use" ở Footes).

### Kết luận cho kết luận 3 của paper

- "Retrieval có thể hại câu nổi tiếng": đúng. 35,6% đến 54,8% số câu bị hại là câu nổi tiếng nhất, nhưng lý do chủ yếu là câu nổi tiếng có nhiều thứ để mất, không phải retrieval tệ hơn ở đó.
- Cái hại đến từ **chất lượng đoạn văn**, như paper giải thích, nhưng có thể chỉ rõ hơn: hoặc đoạn nói về thực thể khác trùng tên hoặc chung một từ trong tên (cả 9 mô hình cùng chép theo), hoặc là một đoạn dùng chung cho hàng trăm chủ thể (một phần ba câu hỏi, tập trung ở producer, screenwriter, author).
- Với adaptive: bỏ tra cứu ở câu nổi tiếng tránh được phần hại lớn nhất (xem `docs/BANG_MO_RONG.md`), nhưng đoạn dùng chung nằm ở mọi mức nổi tiếng (24,6%, 33,4%, 38,6%, 38,9% và 32,4% số câu ở nhóm 1 đến nhóm 5), nên độ nổi tiếng không phân biệt được phần này.

### Giới hạn

- Đọc tay làm trên 38 câu (18 và 20) do một người đọc (Claude Code), chỉ đọc khoảng 230 ký tự đầu của đoạn văn, chưa có người thứ hai kiểm. Các `id` có thể mở lại: 18 câu lấy từ `results/tables/retrieval_harm_questions.csv` (`n_models_hurt` bằng 9); 20 câu bị hại của GPT-J chọn ngẫu nhiên bằng seed 633. 18 câu bị hại ở cả 9 mô hình là các trường hợp cực đoan, không đại diện cho mọi câu bị hại.
- Định nghĩa "đoạn dùng chung" (ít nhất 5 chủ thể khác nhau) là của nhóm đặt ra, không có trong paper.
- recall@1 và "câu trả lời nằm trong đoạn" dùng so khớp chuỗi con như cách chấm, nên có dương tính giả với đáp án ngắn và số hơi cao; cách chấm chặt hơn là việc của #35.
- Chỉ đoạn top-1 của file BM25 do tác giả đăng. Chưa biết Contriever (không còn file) có bị đoạn dùng chung không.
- Nhóm nổi tiếng chia chung cho mọi loại quan hệ, nên một câu ít nổi tiếng trong loại `country` có thể có `s_pop` cao hơn một câu rất nổi tiếng trong loại khác.

Chạy lại: `python scripts/analyze_retrieval_harm.py --bm25 <file BM25>` (ghi 5 file `results/tables/retrieval_harm_*.csv`), kiểm bằng `python scripts/test_analyze_retrieval_harm.py`.

## 4. Đoạn văn có đáp án mà vẫn sai (issue #38)

### Câu hỏi

Có 3.644 câu (25,5%) mà đoạn BM25 top-1 chứa một đáp án đúng. Trong các câu đó BM25 vẫn trả lời sai ở một phần: vì sao đoạn văn đúng mà mô hình vẫn sai?

### Tóm tắt

- Khi đoạn có đáp án, BM25 sai ở **13,0% (Qwen2.5 3B) đến 24,5% (GPT-Neo 1.3B)** số câu, tức 474 đến 891 câu mỗi mô hình. Mô hình đọc đúng 75,5% đến 87,0% số câu như vậy.
- **Phần lớn các câu "sai" này không phải mô hình đọc sai mà là lỗi đo.** Ít nhất 29,3% đến 45,6% số câu sai là đáp án giả hoặc câu trả lời gần đúng về nghĩa mà cách chấm không nhận (hai nhóm tự động bắt được, chỉ là cận dưới). Trong 20 ca GPT-J đọc tay, 15 ca (75%) là lỗi đo hoặc đáp án chuẩn hẹp, chỉ 5 ca (25%) là mô hình thật sự đọc sai.
- Đáp án **không** bị cắt vì prompt dài (xem bên dưới).

Một câu cho slide: *khi đoạn văn đã có đáp án, BM25 vẫn sai 13% đến 25% số câu, nhưng phần lớn là cách chấm quá chặt ("comedy" bị tính sai so với "comedy film") hoặc đáp án chuẩn chọn một nhãn hẹp, không phải mô hình đọc sai.*

### Cách làm

- Tập câu: `answer_in_passage` đúng (3.644 câu, giống nhau ở mọi mô hình vì mọi mô hình dùng chung đoạn top-1), BM25 sai (`is_correct` sai). Dùng recall@1 tính lại, không dùng cờ `has_answer` (xem `docs/BANG_TAI_LAP.md`).
- Chia câu sai thành 4 nhóm bằng quy tắc tự động, xét theo thứ tự: **có đáp án giả** (đáp án chỉ khớp như một phần của chữ khác, ví dụ "pol" trong "Polo"); **gần đúng về nghĩa** (câu trả lời của BM25 là một phần của một đáp án chuẩn hoặc ngược lại, theo tập từ đã chuẩn hóa, ví dụ "comedy" với "comedy film"); **chọn nhầm ứng viên** (câu trả lời có trong đoạn nhưng không phải đáp án); **bỏ qua đoạn** (câu trả lời không có trong đoạn).
- Đọc tay 20 câu GPT-J chọn ngẫu nhiên (seed 633) trong 644 câu có đáp án mà sai, chia nhóm nguyên nhân chi tiết hơn, rồi đối chiếu với nhóm tự động.
- Chạy lại: `python scripts/analyze_answer_present_wrong.py --bm25 <file BM25>` (ghi 3 file `results/tables/answer_present_wrong_*.csv`), kiểm bằng `python scripts/test_analyze_answer_present_wrong.py`.

### Thống kê 1: tỉ lệ theo mô hình và nhóm tự động

| Mô hình | Số câu BM25 sai (trên 3.644) | Tỉ lệ sai | Có đáp án giả (chỉ khớp một phần của chữ) | Đúng hoặc gần đúng về nghĩa | Chọn nhầm một ứng viên khác trong đoạn | Bỏ qua đoạn |
|---|---|---|---|---|---|---|
| GPT-Neo 1.3B | 891 | 24,5% | 12,5% | 16,8% | 48,4% | 22,3% |
| GPT-Neo 2.7B | 742 | 20,4% | 15,1% | 20,9% | 47,4% | 16,6% |
| OPT 1.3B | 885 | 24,3% | 14,2% | 18,8% | 48,1% | 18,9% |
| OPT 2.7B | 750 | 20,6% | 17,5% | 19,1% | 47,3% | 16,1% |
| OPT 6.7B (fp16) | 684 | 18,8% | 17,0% | 20,0% | 42,7% | 20,3% |
| GPT-J 6B (fp16) | 644 | 17,7% | 17,7% | 22,2% | 42,4% | 17,7% |
| Qwen2.5 0.5B | 721 | 19,8% | 16,2% | 16,0% | 50,5% | 17,3% |
| Qwen2.5 1.5B | 545 | 15,0% | 21,7% | 9,5% | 42,4% | 26,4% |
| Qwen2.5 3B | 474 | 13,0% | 22,2% | 23,4% | 41,8% | 12,7% |

- Tỉ lệ sai thấp nhất ở Qwen2.5 3B (13,0%) và Qwen2.5 1.5B (15,0%), cao nhất ở hai mô hình 1.3B (24,3% và 24,5%): mô hình đọc đoạn văn tốt hơn thì ít sai hơn.
- Hai nhóm đầu cộng lại (đáp án giả và gần đúng về nghĩa) là 29,3% đến 45,6% số câu sai, và đây là **cận dưới của lỗi đo**, vì quy tắc tự động không bắt được đáp án giả khi đáp án khớp cả từ (như "USA" trong đoạn về một đài phát thanh) hay đáp án chuẩn hẹp.
- "Chọn nhầm ứng viên" (41,8% đến 50,5%) **không** có nghĩa 40% đến 50% là mô hình đọc sai: nhóm này gom cả các ca đáp án chuẩn hẹp và có đáp án giả, vì mọi câu trả lời có trong đoạn mà không phải đáp án đều rơi vào đây (xem bảng đọc tay: trong 8 ca của nhóm này chỉ 3 ca là chọn nhầm thật).
- Trong số câu sai, 74 đến 132 câu (11,0% đến 22,8%) là câu vanilla từng trả lời đúng.

### Thống kê 2: theo loại quan hệ

GPT-J (tỉ lệ sai khi đoạn có đáp án); cột cuối là khoảng giữa 9 mô hình.

| Loại quan hệ | Câu có đáp án trong đoạn | BM25 sai (GPT-J) | Tỉ lệ sai (GPT-J) | Khoảng giữa 9 mô hình |
|---|---|---|---|---|
| religion | 80 | 51 | 63,7% | 58,8% đến 67,5% |
| genre | 295 | 127 | 43,1% | 24,7% đến 50,2% |
| capital of | 240 | 81 | 33,8% | 17,5% đến 39,6% |
| occupation | 97 | 32 | 33,0% | 22,7% đến 44,3% |
| producer | 102 | 31 | 30,4% | 26,5% đến 34,3% |
| color | 14 | 4 | 28,6% | 14,3% đến 35,7% |
| father | 243 | 63 | 25,9% | 13,6% đến 39,9% |
| screenwriter | 195 | 40 | 20,5% | 17,9% đến 33,3% |
| mother | 58 | 8 | 13,8% | 10,3% đến 48,3% |
| country | 709 | 93 | 13,1% | 11,1% đến 22,1% |
| composer | 107 | 13 | 12,1% | 6,5% đến 19,6% |
| place of birth | 319 | 37 | 11,6% | 9,1% đến 28,2% |
| director | 140 | 11 | 7,9% | 3,6% đến 24,3% |
| capital | 355 | 25 | 7,0% | 5,9% đến 14,9% |
| author | 293 | 12 | 4,1% | 2,7% đến 8,5% |
| sport | 397 | 16 | 4,0% | 1,8% đến 5,3% |

`religion` có tỉ lệ sai cao nhất ở cả 9 mô hình (58,8% đến 67,5%, cao hơn mọi loại quan hệ khác ở mọi mô hình). `genre` cao thứ hai ở GPT-J (43,1%; từ 24,7% đến 50,2% ở 9 mô hình) và là loại có nhiều câu sai nhất: 19,7% số câu sai của GPT-J, `genre` cộng `religion` là 27,6%. Gợi ý (chưa kiểm cho `religion`): đây là các loại quan hệ có nhiều cách gọi đúng cho cùng một ý (comedy, comedy film), nên cách chấm so chuỗi dễ chấm sai; trong 20 ca đọc tay, 9 ca là `genre` và 5 trong 7 ca "gần đúng về nghĩa" là `genre`. `author` và `sport`, nơi đáp án thường là một tên riêng hoặc một từ, sai ít nhất (4,1% và 4,0% ở GPT-J).

### Đọc tay: 20 ca GPT-J

| Câu hỏi | Đáp án chuẩn | BM25 trả lời | Nhóm (đọc tay) |
|---|---|---|---|
| What genre is A Painted House? (`308097`) | legal thriller | historical fiction | Có đáp án giả |
| What is Marco Ameglio's occupation? (`900521`) | politician | music critic | Có đáp án giả |
| In what country is Arthur? (`3381798`) | United States of America | England | Có đáp án giả |
| What is the religion of Spring Hill College? (`4272302`) | Catholic Church | Methodist | Có đáp án giả |
| Who is the father of Stephen of England? (`4582156`) | Stephen | Paul McGann | Có đáp án giả |
| Who was the screenwriter for Keep Fit? (`4830188`) | Anthony Kimmins | John Boulting | Bỏ qua đoạn |
| What is the capital of Orotina Canton? (`6370830`) | Orotina | San José | Bỏ qua đoạn |
| Who was the producer of Pin Up Girl? (`2576512`) | 20th Century Studios | William LeBaron | Đáp án chuẩn hẹp |
| What genre is Afsporet? (`3237038`) | film noir | Psychological thriller | Đáp án chuẩn hẹp |
| What genre is Frans Floris I? (`4097458`) | Northern Renaissance | history painting | Đáp án chuẩn hẹp |
| What genre is City Slickers? (`144090`) | comedy film | comedy | Đúng hoặc gần đúng về nghĩa, chấm sai |
| What genre is Kei Yasuda? (`505960`) | J-pop | pop | Đúng hoặc gần đúng về nghĩa, chấm sai |
| Who was the director of Maine Pyar Kiya? (`1303963`) | Sooraj R. Barjatya | Sooraj Barjatya | Đúng hoặc gần đúng về nghĩa, chấm sai |
| What genre is Abandon? (`3205878`) | Christian rock | Rock | Đúng hoặc gần đúng về nghĩa, chấm sai |
| Who is the author of Ang Singsing ng Dalagang Marmol? (`3337761`) | Isabelo de los Reyes | Isabelo Florentino de los Reyes | Đúng hoặc gần đúng về nghĩa, chấm sai |
| What genre is Blackbirds at Bangpleng? (`3521531`) | horror film | Science fiction horror | Đúng hoặc gần đúng về nghĩa, chấm sai |
| What genre is Value for Money? (`6092302`) | comedy film | comedy | Đúng hoặc gần đúng về nghĩa, chấm sai |
| What is Hanover the capital of? (`1188844`) | Lower Saxony | Germany | Chọn nhầm trong đoạn |
| Who is the father of Princess Mary, Duchess of Gloucester and Edinburgh? (`1830760`) | George III of Great Britain | Prince William Frederick, Duke of Gloucester and Edinburgh | Chọn nhầm trong đoạn |
| What genre is Just Dance 4? (`2849350`) | music video game | Dance | Chọn nhầm trong đoạn |

| Nhóm nguyên nhân | Số ca | Khoảng tin cậy 95% | Ý nghĩa |
|---|---|---|---|
| Đúng hoặc gần đúng về nghĩa, chấm sai | 7 | 18% đến 57% | Câu trả lời đúng về ý nhưng không chứa chuỗi đáp án chuẩn: thiếu chữ ("comedy" so với "comedy film", "pop" so với "Japanese pop", "Rock" so với "Christian rock"), thiếu chữ viết tắt ("Sooraj Barjatya" so với "Sooraj R. Barjatya"), tên đầy đủ hơn ("Isabelo Florentino de los Reyes") |
| Có đáp án giả | 5 | 11% đến 47% | Đáp án khớp nhưng đoạn không trả lời câu hỏi: "pol" nằm trong "Marco **Pol**o"; "Stephen" nằm trong "Stephen McGann"; "Church" nằm trong "Spring Hill Methodist Church" (một nơi khác); "USA" trong đoạn về một đài phát thanh ở Kentucky; "legal thriller" xuất hiện trong câu nói tác phẩm này **nằm ngoài** thể loại đó |
| Đáp án chuẩn hẹp | 3 | 5% đến 36% | Câu trả lời hợp lý theo đoạn nhưng đáp án chuẩn chọn nhãn khác: đoạn Pin Up Girl ghi "produced by William LeBaron" còn đáp án chuẩn là hãng phim 20th Century Fox; Afsporet được gọi là "thriller" và cũng là "film noir", đáp án chuẩn chỉ có "film noir"; Frans Floris vừa vẽ "history paintings" vừa thuộc "Northern Renaissance" |
| Chọn nhầm trong đoạn | 3 | 5% đến 36% | Đáp án nằm trong đoạn nhưng mô hình chọn một ứng viên khác: Hanover chọn "Germany" thay cho "Lower Saxony" trong cùng một câu; Princess Mary chọn chồng ("Prince William Frederick") thay cho cha ("King George III"); Just Dance 4 chọn "Dance" (một chữ trong tên, đoạn thì nói về Just Dance 3) |
| Bỏ qua đoạn | 2 | 3% đến 30% | Câu trả lời không có trong đoạn dù đoạn có đáp án: Keep Fit (đoạn nêu Anthony Kimmins là đạo diễn, không phải biên kịch; mô hình trả lời một tên không có trong đoạn); Orotina (đoạn viết thẳng "Orotina is the capital city of the canton of Orotina", mô hình trả lời San José) |

- Chỉ **5 trên 20 (25%, khoảng tin cậy 11% đến 47%)** là mô hình thật sự đọc sai (chọn nhầm và bỏ qua đoạn). **15 trên 20 (75%, khoảng 53% đến 89%)** là lỗi đo hoặc đáp án chuẩn hẹp. Hai ước lượng này rất rộng vì chỉ có 20 ca.
- Nhóm tự động khớp với đọc tay ở nhóm gần đúng về nghĩa: quy tắc tự động bắt đúng 6 trên 7 ca và cả 6 ca nó bắt đều đúng là loại này (không bắt nhầm). Cả 3 ca chọn nhầm đều ở nhóm "chọn nhầm ứng viên", cả 2 ca bỏ qua đoạn đều ở nhóm "bỏ qua đoạn".

### Một nguyên nhân đã loại: đáp án bị cắt vì prompt dài

Code gốc cắt prompt từ **bên trái** (giữ phần cuối) ở 2.048 token. Đoạn văn chính nằm sát cuối prompt, ngay trước câu hỏi, nên chỉ các ví dụ mẫu đầu tiên mới bị cắt. Đoạn top-1 dài nhất có 5.322 ký tự, đo bằng tokenizer GPT-2 chỉ khoảng 1.100 đến 1.230 token (kể cả dòng câu hỏi), thấp hơn hẳn 2.033 token dành cho prompt. Vậy đoạn văn chính không bao giờ bị cắt. (Tokenizer GPT-2 là xấp xỉ cho GPT-Neo và OPT; chưa đo với tokenizer của Qwen.)

### Kết luận

- Khi đoạn BM25 đã có đáp án, mô hình đọc đúng 75,5% đến 87,0% số câu; mô hình mạnh hơn đọc tốt hơn (Qwen2.5 3B sai 13,0%).
- Phần "sai" còn lại **phần lớn là lỗi đo, không phải lỗi đọc**: ít nhất 29,3% đến 45,6% (đo tự động trên mọi câu) và khoảng 75% (đọc tay 20 ca, chưa chắc chắn). Nguyên nhân chính là cách chấm so chuỗi con: nhận cả đáp án giả ("pol" trong "Polo") và không nhận câu trả lời gần đúng ("comedy" so với "comedy film"). Việc đo lại bằng cách chấm chặt hơn là của #35.
- Số câu bị hại ở mục 3 và recall@1 ở `docs/BANG_TAI_LAP.md` cũng chịu cùng điểm yếu này. Cách làm đúng nhất là chấm lại bằng cách chấm chặt hơn rồi tính lại các bảng đó.
- Lỗi đọc thật còn lại thuộc hai kiểu: chọn nhầm một thực thể gần đáp án trong cùng đoạn (vợ chồng, nước và bang), hoặc bỏ qua đoạn và trả lời từ trí nhớ.

### Giới hạn

- Đọc tay chỉ làm trên 20 ca của một mô hình (GPT-J) và do một người đọc (Claude Code), chưa ai kiểm lại. Khoảng tin cậy rất rộng.
- Mẫu có **9 trên 20 câu thuộc loại `genre`**, nhiều hơn mức kỳ vọng khoảng 4 (xác suất gặp từ 9 trở lên là 0,9%), và `genre` chiếm 5 trên 7 ca "gần đúng về nghĩa". Vì vậy tỉ lệ 35% của nhóm đó trong mẫu có thể bị thổi lên; số tự động (9,5% đến 23,4% trên mọi câu, là cận dưới) đáng tin hơn về mức độ.
- Nhóm tự động là xấp xỉ: "gần đúng" dựa trên tập từ, bỏ sót các ca như "Science fiction horror" so với "horror film"; "chọn nhầm ứng viên" lẫn cả đáp án chuẩn hẹp và đáp án giả.
- "Có đáp án" dựa trên `answer_in_passage` nên có dương tính giả (xem trên); quy tắc chấm `is_correct` cũng có điểm yếu tương tự.
- Chưa phân tích theo độ nổi tiếng hay theo độ dài đoạn.
