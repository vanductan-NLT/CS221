# Phân tích (10%)

Mục 1, 2, 4 làm ở issue #35, #36, #38. File này viết tiếp khi các issue đó xong.

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
