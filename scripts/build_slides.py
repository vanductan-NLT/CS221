"""Tạo bộ slide thuyết trình 15 phút cho đồ án cuối kỳ CS221 (PopQA & Adaptive Retrieval).

Chạy:
    python scripts/build_slides.py
Đầu ra:
    slides/CS221_Adaptive_Retrieval.pptx
"""
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figs"
OUT_PPTX = ROOT / "slides" / "CS221_Adaptive_Retrieval.pptx"

# Bảng màu học thuật hiện đại (Clean Academic Navy + Coral + Teal)
NAVY = RGBColor(0x16, 0x2B, 0x4D)
NAVY_LIGHT = RGBColor(0x23, 0x42, 0x70)
SLATE = RGBColor(0x33, 0x41, 0x55)
MUTED = RGBColor(0x64, 0x74, 0x8B)
BG = RGBColor(0xF8, 0xFA, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xE2, 0xE8, 0xF0)
CORAL = RGBColor(0xD9, 0x4F, 0x3D)
CORAL_BG = RGBColor(0xFD, 0xED, 0xEB)
TEAL = RGBColor(0x0D, 0x94, 0x88)
TEAL_BG = RGBColor(0xE6, 0xF7, 0xF5)
AMBER = RGBColor(0xD9, 0x77, 0x06)
AMBER_BG = RGBColor(0xFE, 0xF3, 0xC7)
BLUE_BG = RGBColor(0xEB, 0xF2, 0xFA)
ROW_ALT = RGBColor(0xF1, 0xF5, 0xF9)

FONT = "Segoe UI"
TOTAL_SLIDES = 15


def set_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color, border_color=None, border_pt=1.0, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(width), Inches(height))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_color
    if border_color:
        shp.line.color.rgb = border_color
        shp.line.width = Pt(border_pt)
    else:
        shp.line.fill.background()
    if radius and len(shp.adjustments) > 0:
        shp.adjustments[0] = 0.06
    return shp


def add_textbox(slide, left, top, width, height, text="", size=14, bold=False, color=SLATE,
                align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    if text:
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def add_bullets(slide, left, top, width, height, items, size=13.5, space_after=6, color=SLATE, bullet="• "):
    """items: list of str or tuple(bold_prefix, rest_text)."""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.18
        if isinstance(item, tuple):
            prefix, rest = item
            r0 = p.add_run()
            r0.text = bullet + prefix
            r0.font.name = FONT
            r0.font.size = Pt(size)
            r0.font.bold = True
            r0.font.color.rgb = NAVY
            r1 = p.add_run()
            r1.text = rest
            r1.font.name = FONT
            r1.font.size = Pt(size)
            r1.font.color.rgb = color
        else:
            r = p.add_run()
            r.text = bullet + item
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.color.rgb = color
    return tb


def add_header(slide, title, subtitle, section_tag, tag_color, slide_num):
    set_bg(slide, BG)
    # Thanh nhấn trên cùng
    add_rect(slide, 0, 0, 13.333, 0.08, tag_color)
    # Badge phần
    badge = add_rect(slide, 0.55, 0.25, 1.95, 0.30, tag_color, radius=True)
    badge.text_frame.word_wrap = False
    badge.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = badge.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = section_tag
    r.font.name = FONT
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = WHITE

    add_textbox(slide, 2.65, 0.19, 10.1, 0.42, title, size=22, bold=True, color=NAVY)
    if subtitle:
        add_textbox(slide, 0.55, 0.63, 12.2, 0.32, subtitle, size=12.5, color=MUTED)
    # Đường kẻ phân cách mảnh
    add_rect(slide, 0.55, 0.98, 12.23, 0.015, BORDER)
    # Footer
    add_textbox(slide, 0.55, 7.12, 9.5, 0.28,
                "CS221 · Nhập môn NLP (UIT) · Tái lập & Mở rộng 'When Not to Trust Language Models' (ACL 2023)",
                size=9.5, color=MUTED)
    add_textbox(slide, 11.6, 7.12, 1.2, 0.28, f"{slide_num} / {TOTAL_SLIDES}",
                size=9.5, bold=True, color=NAVY, align=PP_ALIGN.RIGHT)


def add_kpi_card(slide, left, top, width, height, value, label, accent=NAVY, bg=WHITE):
    add_rect(slide, left, top, width, height, bg, border_color=BORDER, border_pt=1.0, radius=True)
    add_rect(slide, left, top, 0.08, height, accent)
    add_textbox(slide, left + 0.18, top + 0.08, width - 0.25, 0.42, value, size=20, bold=True, color=accent)
    add_textbox(slide, left + 0.18, top + 0.50, width - 0.25, height - 0.55, label, size=10.5, color=SLATE)


def add_card(slide, left, top, width, height, heading, items, accent=NAVY, bg=WHITE, size=12.5, space_after=5):
    add_rect(slide, left, top, width, height, bg, border_color=BORDER, border_pt=1.0, radius=True)
    add_rect(slide, left, top, width, 0.06, accent)
    add_textbox(slide, left + 0.16, top + 0.12, width - 0.32, 0.34, heading, size=14, bold=True, color=accent)
    add_bullets(slide, left + 0.16, top + 0.48, width - 0.32, height - 0.56, items, size=size, space_after=space_after)


def add_picture_fit(slide, img_path: Path, left, top, max_w, max_h):
    with Image.open(img_path) as im:
        w_px, h_px = im.size
    ratio = w_px / h_px
    box_ratio = max_w / max_h
    if ratio > box_ratio:
        w = max_w
        h = max_w / ratio
        x = left
        y = top + (max_h - h) / 2
    else:
        h = max_h
        w = max_h * ratio
        x = left + (max_w - w) / 2
        y = top
    # Khung trắng gọn bao quanh hình
    add_rect(slide, x - 0.05, y - 0.05, w + 0.10, h + 0.10, WHITE, border_color=BORDER, border_pt=1.0)
    slide.shapes.add_picture(str(img_path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def add_styled_table(slide, left, top, width, height, headers, rows, col_widths=None,
                     font_size=10.5, highlight_rows=None):
    highlight_rows = set(highlight_rows or [])
    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl_shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top), Inches(width), Inches(height))
    tbl = tbl_shape.table
    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Inches(cw)

    # Header
    for c, h in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = h
        r.font.name = FONT
        r.font.size = Pt(font_size)
        r.font.bold = True
        r.font.color.rgb = WHITE

    # Body
    for r_idx, row in enumerate(rows):
        is_hl = r_idx in highlight_rows
        bg_c = CORAL_BG if is_hl else (WHITE if r_idx % 2 == 0 else ROW_ALT)
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_c
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(font_size)
            r.font.bold = is_hl or (c_idx == 0)
            r.font.color.rgb = CORAL if (is_hl and c_idx == 0) else (NAVY if c_idx == 0 else SLATE)
    return tbl_shape


def set_notes(slide, text):
    notes = slide.notes_slide.notes_text_frame
    notes.text = text.strip()


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TRANG BÌA
    # =========================================================================
    s = prs.slides.add_slide(blank)
    set_bg(s, NAVY)
    add_rect(s, 0, 0, 0.35, 7.5, CORAL)
    add_rect(s, 0.35, 0, 0.12, 7.5, TEAL)

    add_textbox(s, 1.0, 0.95, 11.2, 0.4,
                "ĐỒ ÁN CUỐI KỲ · CS221 — NHẬP MÔN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (UIT)",
                size=13, bold=True, color=RGBColor(0x93, 0xC5, 0xFD))
    add_textbox(s, 1.0, 1.55, 11.4, 1.5,
                "When Not to Trust Language Models:\nTái lập, Mở rộng & Phân tích Adaptive Retrieval",
                size=34, bold=True, color=WHITE)
    add_textbox(s, 1.0, 3.25, 11.2, 0.95,
                "Nghiên cứu lại bài báo Mallen et al. (ACL 2023, Long Paper) trên trọn bộ 14.267 câu PopQA,\n"
                "mở rộng sang họ mô hình Qwen2.5 (0.5B–3B), Adaptive theo độ tự tin và phân tích 4 cơ chế lỗi.",
                size=15.5, color=RGBColor(0xE2, 0xE8, 0xF0))

    # 3 hộp tóm tắt trọng số
    cards_1 = [
        ("60% · TÁI LẬP", "6 mô hình gốc × 14.267 câu\nKhớp paper: MAD = 0,18 điểm %", TEAL),
        ("30% · MỞ RỘNG", "Qwen2.5 (0.5B–3B) + Kiểm định\nAdaptive theo độ tự tin (conf)", CORAL),
        ("10% · PHÂN TÍCH", "Chấm chặt · 16 quan hệ\nĐoạn dùng chung (33,6%) · Lỗi đo", AMBER),
    ]
    for i, (t, desc, col) in enumerate(cards_1):
        x = 1.0 + i * 3.85
        add_rect(s, x, 4.65, 3.55, 1.35, NAVY_LIGHT, border_color=col, border_pt=1.5, radius=True)
        add_textbox(s, x + 0.18, 4.78, 3.2, 0.35, t, size=13, bold=True, color=col)
        add_textbox(s, x + 0.18, 5.18, 3.2, 0.72, desc, size=12, color=WHITE)

    add_textbox(s, 1.0, 6.45, 11.2, 0.55,
                "Thời lượng trình bày: 15 phút  |  Repo tái lập: github.com/vanductan-NLT/CS221  |  9/9 bộ kiểm thử PASS",
                size=11.5, color=RGBColor(0x94, 0xA3, 0xB8))
    set_notes(s, (
        "Slide 1 (0,25 phút): Giới thiệu đề tài đồ án CS221. Nhóm tái lập paper 'When Not to Trust Language Models' "
        "(ACL 2023) trên toàn bộ 14.267 câu PopQA, mở rộng sang họ Qwen2.5 và Adaptive theo độ tự tin, đồng thời phân tích 4 nhóm lỗi chính."
    ))

    # =========================================================================
    # SLIDE 2: BÀI TOÁN & CÁC KHÁI NIỆM CỐT LÕI
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Bài toán & Các khái niệm cốt lõi cần nắm",
               "Khi nào nên tin trí nhớ tham số của mô hình ngôn ngữ, khi nào cần tra cứu thêm kiến thức ngoài?",
               "60% · BÀI TOÁN", NAVY, 2)

    add_card(s, 0.55, 1.15, 6.0, 2.75, "1. Hai nguồn tri thức của LLM", [
        ("Trí nhớ tham số (Parametric — Vanilla): ", "Kiến thức mô hình tự ghi nhớ trong trọng số sau quá trình pre-train. Nhanh, rẻ, nhưng dễ bịa (hallucinate) với thực thể hiếm."),
        ("Trí nhớ ngoài (Non-parametric — RAG): ", "Dùng bộ tìm kiếm lấy 1 đoạn Wikipedia dán vào trước câu hỏi. Cứu được câu hiếm nhưng chậm hơn và có thể gây nhiễu."),
        ("Ví dụ trực quan: ", "“Thủ đô Louisiana?” → LLM tự biết (Baton Rouge). “Nghề nghiệp Kathy Saltzman?” → LLM không nhớ, cần tra Wikipedia."),
    ], accent=NAVY, size=12.2)

    add_card(s, 6.78, 1.15, 6.0, 2.75, "2. Ba phương pháp bổ trợ trong bài báo", [
        ("BM25 (Sparse / Từ khóa): ", "Thuật toán tìm kiếm cổ điển khớp từ khóa hiếm, không cần train. Nhanh nhưng dễ nhầm thực thể trùng tên."),
        ("Contriever (Dense / Ngữ nghĩa): ", "Bộ tìm kiếm mạng nơ-ron so khớp vector ngữ nghĩa (recall@1 đạt 42% trên PopQA, mạnh hơn BM25)."),
        ("GenRead (Tự sinh ngữ cảnh): ", "Không tra Wikipedia — bảo chính LLM tự viết đoạn văn nền (~150 token) rồi tự đọc để trả lời. Vẫn 100% trí nhớ tham số."),
    ], accent=TEAL, size=12.2)

    add_card(s, 0.55, 4.10, 12.23, 2.82, "3. Ba câu hỏi nghiên cứu (RQ1 – RQ3) bài báo giải quyết", [
        ("RQ1 — Trí nhớ tham số nhớ được bao nhiêu và phụ thuộc yếu tố nào? ", "Đo khả năng nhớ theo độ nổi tiếng thực thể (s_pop) và loại quan hệ; kiểm tra xem tăng kích thước mô hình có cứu được kiến thức đuôi dài (long-tail) không."),
        ("RQ2 — Trí nhớ ngoài (BM25, Contriever) và GenRead giúp được tới đâu? ", "Định lượng mức tăng ở câu ít nổi tiếng và phát hiện mặt trái: đoạn văn sai có thể đánh lừa mô hình trả lời sai câu nó vốn tự biết."),
        ("RQ3 — Có thể kết hợp thích ứng (Adaptive Retrieval) không? ", "Chỉ tra cứu khi câu hỏi rơi vào vùng mô hình ít nhớ → vừa tăng độ chính xác vừa giảm chi phí/độ trễ suy luận."),
    ], accent=CORAL, bg=WHITE, size=12.5)
    set_notes(s, (
        "Slide 2 (1,5 phút): Giải thích trực giác bài toán và 3 khái niệm BM25, Contriever, GenRead. "
        "Nhấn mạnh sự đánh đổi: Vanilla nhanh nhưng yếu ở sự kiện hiếm; RAG cứu sự kiện hiếm nhưng chậm và có thể gây nhiễu."
    ))

    # =========================================================================
    # SLIDE 3: BỘ DỮ LIỆU POPQA & 4 KẾT LUẬN CỦA PAPER
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Bộ dữ liệu PopQA & 4 kết luận chính của bài báo",
               "Vì sao cần PopQA thay cho Natural Questions, các tập dữ liệu dùng làm gì và bài báo chứng minh điều gì?",
               "60% · DỮ LIỆU", NAVY, 3)

    add_card(s, 0.55, 1.15, 6.25, 5.75, "Vai trò của từng tập dữ liệu trong paper", [
        ("PopQA (14.267 câu, 16 loại quan hệ — TẬP CHÍNH): ",
         "Tác giả lấy bộ ba (Chủ thể, Quan hệ, Đáp án) từ Wikidata rồi chuyển thành câu hỏi theo mẫu. Mỗi câu gắn sẵn độ nổi tiếng s_pop = lượt xem Wikipedia/tháng."),
        ("Phân tầng Q1 – Q5 (5 nhóm độ nổi tiếng): ",
         "Chia đều 14.267 câu (~2.853 câu/nhóm): Q1 là 20% thực thể hiếm nhất (long-tail), tăng dần tới Q5 là 20% nổi tiếng nhất (head)."),
        ("EntityQuestions (tập kiểm chứng chéo): ",
         "Bộ QA đuôi dài có sẵn (dùng số hyperlink Wikipedia làm proxy) để kiểm tra kết luận có tổng quát ngoài PopQA không."),
        ("Natural Questions (NQ-open — chỉ vẽ Hình 3): ",
         "Không dùng để đánh giá mô hình; chỉ so phân bố để chỉ ra NQ toàn hỏi thực thể quá nổi tiếng, che khuất điểm yếu long-tail."),
        ("Cách chấm điểm gốc: ", "Đúng nếu 1 đáp án chuẩn trong possible_answers xuất hiện dưới dạng chuỗi con trong câu trả lời."),
    ], accent=NAVY, size=12.0)

    add_card(s, 7.05, 1.15, 5.73, 5.75, "4 kết luận cốt lõi paper chứng minh", [
        ("KL1 — Càng nổi tiếng càng hay đúng: ",
         "Độ chính xác vanilla tương quan dương rõ rệt với log(s_pop) ở hầu hết các loại quan hệ; các quan hệ dễ đoán (country, sport) có tương quan thấp hơn."),
        ("KL2 — Mô hình to hơn chủ yếu giỏi hơn ở câu nổi tiếng: ",
         "Tăng cỡ từ 1.3B → 20B → GPT-3 (~175B) tăng mạnh ở vùng head, nhưng trên 4.000 câu ít nổi nhất gần như giậm chân tại chỗ (GPT-J 6B: 15–16%, GPT-3: 19%)."),
        ("KL3 — Retrieval giúp long-tail, nhưng có thể HẠI câu nổi tiếng: ",
         "Bảng 1 paper (GPT-3 + Contriever): 17% câu được cứu (recall@1 = 0,88), nhưng 10% câu vốn đúng bị làm sai do đoạn văn kém (recall@1 = 0,14)."),
        ("KL4 — Adaptive Retrieval tốt bằng/hơn cả hai cách & rẻ hơn: ",
         "Chỉ tra cứu khi cần giúp GPT-3 + BM25 giảm còn ~40–45% lượt tra cứu mà vẫn đạt độ chính xác cao nhất (39,5%; với GenRead/Contriever đạt 46,5%)."),
    ], accent=CORAL, bg=WHITE, size=12.0)
    set_notes(s, (
        "Slide 3 (1,0 phút): Trình bày lý do tác giả tạo PopQA (NQ quá nổi tiếng), giải thích khái niệm s_pop và 5 tầng Q1–Q5, "
        "và tóm gọn 4 kết luận mà nhóm sẽ kiểm chứng lại."
    ))

    # =========================================================================
    # SLIDE 4: PHƯƠNG PHÁP ADAPTIVE RETRIEVAL & CÔNG THỨC NGƯỠNG
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Phương pháp Adaptive Retrieval & Cách tính ngưỡng",
               "Quy trình chọn ngưỡng độ nổi tiếng (popularity threshold) theo Mục 6.1 & Appendix B của bài báo",
               "60% · PHƯƠNG PHÁP", NAVY, 4)

    # Sơ đồ luồng trực quan 3 bước
    steps_4 = [
        ("BƯỚC 1 · CHIA DỮ LIỆU", "Tách ngẫu nhiên 75% Train / 25% Test trên PopQA.\nLặp 100 lần (seed = 633) rồi lấy trung bình.", NAVY),
        ("BƯỚC 2 · HỌC NGƯỠNG THEO QUAN HỆ", "Với mỗi loại quan hệ r (16 loại), chia log(s_pop) thành 10 bins → 11 ứng viên ngưỡng i ∈ {0..10}.", TEAL),
        ("BƯỚC 3 · QUYẾT ĐỊNH TRÊN TEST", "Nếu bin(x) < i* → dùng Retrieval (BM25).\nNếu bin(x) ≥ i* → dùng Vanilla (trí nhớ LLM).", CORAL),
    ]
    for i, (st, desc, col) in enumerate(steps_4):
        x = 0.55 + i * 4.18
        add_rect(s, x, 1.18, 3.88, 1.65, WHITE, border_color=col, border_pt=1.5, radius=True)
        add_rect(s, x, 1.18, 3.88, 0.36, col)
        add_textbox(s, x + 0.12, 1.21, 3.65, 0.30, st, size=11.5, bold=True, color=WHITE)
        add_textbox(s, x + 0.14, 1.62, 3.60, 1.12, desc, size=11.8, color=SLATE)
        if i < 2:
            add_textbox(s, x + 3.89, 1.75, 0.28, 0.4, "→", size=20, bold=True, color=MUTED, align=PP_ALIGN.CENTER)

    # Hộp công thức tối ưu
    add_rect(s, 0.55, 3.05, 12.23, 1.55, BLUE_BG, border_color=NAVY, border_pt=1.2, radius=True)
    add_textbox(s, 0.78, 3.15, 11.8, 0.32,
                "Công thức tìm kiếm vét cạn (Brute-force search) ngưỡng tối ưu i* trên tập train cho từng loại quan hệ r:",
                size=13, bold=True, color=NAVY)
    add_textbox(s, 0.78, 3.55, 11.8, 0.48,
                "i*  =  argmax_{i ∈ {0..10}}  [  Σ_{b < i} Đúng_Retrieval(b)  +  Σ_{b ≥ i} Đúng_Vanilla(b)  ]  /  N_{train, r}",
                size=15, bold=True, color=CORAL, align=PP_ALIGN.CENTER)
    add_textbox(s, 0.78, 4.12, 11.8, 0.36,
                "Quy ước biên:  i* = 10 nghĩa là luôn tra cứu mọi câu của quan hệ r;   i* = 0 nghĩa là không bao giờ tra cứu (tin hoàn toàn vào Vanilla).",
                size=11.5, color=SLATE, align=PP_ALIGN.CENTER)

    add_card(s, 0.55, 4.80, 6.0, 2.10, "Vì sao phải đặt ngưỡng riêng theo từng quan hệ?", [
        ("Độ khó khác biệt lớn: ", "Quan hệ capital/country mô hình tự biết nhiều nên ngưỡng thấp (ít tra cứu); quan hệ director/producer mô hình yếu nên ngưỡng cao (tra cứu gần như toàn bộ)."),
        ("Khớp thuật toán gốc: ", "Hàm adaptive_eval trong scripts/aggregate.py được kiểm thử khớp 100% số học với notebook gốc của tác giả."),
    ], accent=TEAL, size=11.8)

    add_card(s, 6.78, 4.80, 6.0, 2.10, "Ưu điểm & Hạn chế của Adaptive gốc", [
        ("Ưu điểm: ", "Không cần huấn luyện lại mô hình, chỉ cần chạy ngưỡng trên 2 kết quả đã có; giảm mạnh thời gian chạy & chi phí API với model lớn."),
        ("Hạn chế mở ra Hướng C: ", "Ngoài thực tế người dùng đặt câu hỏi bất kỳ thì hệ thống không biết sẵn s_pop → nhóm thử thay bằng độ tự tin conf."),
    ], accent=AMBER, size=11.8)
    set_notes(s, (
        "Slide 4 (1,25 phút): Giải thích rõ công thức brute-force tìm ngưỡng popularity threshold trong paper "
        "(chia 10 bins log s_pop cho từng quan hệ, tối ưu trên 75% train, đo trên 25% test, lặp 100 lần). "
        "Nhấn mạnh hạn chế không biết trước s_pop ngoài đời để mở đường sang Mở rộng C ở phần sau."
    ))

    # =========================================================================
    # SLIDE 5: THIẾT LẬP TÁI LẬP, PHẠM VI & LỖI CODE GỐC
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Thiết lập tái lập, Phạm vi & Lỗi phát hiện ở code gốc",
               "Tái lập đầy đủ 14.267 câu trên GPU miễn phí (Kaggle T4 ×2) và kiểm chứng 10 lỗi trong mã nguồn công khai",
               "60% · THIẾT LẬP", NAVY, 5)

    add_card(s, 0.55, 1.15, 5.95, 2.85, "Phạm vi nhóm chạy lại (đủ 14.267 câu)", [
        ("6 mô hình của paper: ", "GPT-Neo 1.3B, 2.7B; OPT 1.3B, 2.7B (fp32); GPT-J 6B, OPT 6.7B (fp16 để vừa VRAM T4 15 GB)."),
        ("Các chế độ: ", "Vanilla (lưu thêm độ tự tin conf), BM25 (top-1 Wikipedia), GenRead (GPT-Neo 1.3B & OPT 1.3B), Adaptive (75/25 × 100 lần)."),
        ("Giữ nguyên cấu hình gốc: ", "Template 'Q: {q} A:', 15-shot khác loại quan hệ, greedy decoding max_new_tokens=15, cắt trái 2.048 token, seed=633."),
    ], accent=NAVY, size=11.8)

    add_card(s, 0.55, 4.18, 5.95, 2.72, "Phần không tái lập được (nêu rõ phạm vi)", [
        ("GPT-3 (davinci-002/003): ", "OpenAI đã ngừng cung cấp từ đầu 2024 → trích số paper tham khảo."),
        ("Contriever: ", "Tác giả xác nhận đã mất thư mục Drive chứa file kết quả (issue #5) → dùng file BM25 chính thức tác giả đăng lại."),
        ("OPT 13B, GPT-NeoX 20B & EntityQuestions: ", "13B/20B cần GPU 32 GB (V100); ưu tiên hoàn thành trọn 14.267 câu PopQA cho 9 mô hình."),
    ], accent=MUTED, size=11.8)

    add_card(s, 6.75, 1.15, 6.03, 5.75, "Các lỗi quan trọng phát hiện trong code gốc (upstream/)", [
        ("Lỗi 1 (nghiêm trọng nhất) — Mất toàn bộ đoạn văn ở 15 ví dụ mẫu BM25: ",
         "Câu chính tra bằng nguyên văn (khớp 14.267/14.267), còn 15 ví dụ mẫu tra bằng chữ thường bỏ '?' nên khớp 0/14.267 → code âm thầm in 'missing retrieval' rồi bỏ qua! Nhóm sửa để ví dụ mẫu kèm đoạn văn top-1."),
        ("Lỗi 2 — Ghép Adaptive theo chuỗi question: ",
         "PopQA có 881 câu hỏi trùng văn bản (phủ 2.080 dòng, ví dụ hai người trùng tên). Ghép theo text làm lệch dòng vanilla/BM25 → nhóm sửa ghép theo id."),
        ("Lỗi 3 — Cờ hasanswer có sẵn đánh giá thiếu recall@1: ",
         "Cờ gốc trong file BM25 chỉ báo 7,86%; tính lại đúng quy tắc chấm ra 25,54% (bỏ sót 2.528/3.644 câu)."),
        ("Lỗi 4 — Các lỗi chạy hỏng khác: ",
         "Thiếu thư viện trong requirements.txt, sai tên cờ --ret_file vs --ret_path, lỗi --continue_from đọc sep='\\t' trên file CSV dấu phẩy, has_answer của GenRead đếm cả prompt."),
    ], accent=CORAL, bg=WHITE, size=11.8)
    set_notes(s, (
        "Slide 5 (1,25 phút): Trình bày minh bạch phạm vi chạy lại và các phần không chạy lại được. "
        "Nhấn mạnh công sức kiểm chứng code gốc: phát hiện lỗi 0/14.267 ví dụ mẫu BM25 mất đoạn văn và lỗi 881 câu trùng văn bản."
    ))

    # =========================================================================
    # SLIDE 6: KẾT QUẢ TÁI LẬP — PAPER VS NHÓM
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Kết quả tái lập: Paper gốc (ACL 2023) vs Nhóm chạy lại",
               "Số liệu paper trích chính xác từ tọa độ vector PDF (Hình 7, 9, 10, 13; sai số trục < 1e-10 pt)",
               "60% · KẾT QUẢ", NAVY, 6)

    add_picture_fit(s, FIGS / "paper_vs_repro.png", 0.55, 1.12, 12.23, 3.05)

    # 3 KPI cards + bảng gọn dưới
    add_kpi_card(s, 0.55, 4.30, 3.85, 0.92, "0,18 điểm %", "Sai lệch tuyệt đối trung bình (MAD) trên cả 16 ô so sánh", TEAL)
    add_kpi_card(s, 4.74, 4.30, 3.85, 0.92, "25,55% → 25,56%", "GPT-J 6B Adaptive: khớp paper gần tuyệt đối (+0,01pp)", NAVY)
    add_kpi_card(s, 8.93, 4.30, 3.85, 0.92, "15,22% vs 15–16%", "GPT-J 6B trên 4.000 câu ít nổi nhất (khớp Mục 4.2 paper)", CORAL)

    headers_6 = ["Mô hình (14.267 câu)", "Vanilla (Paper→Nhóm)", "BM25 (Paper→Nhóm)", "GenRead (Paper→Nhóm)", "Adaptive (Paper→Nhóm)", "Tỉ lệ tra cứu"]
    rows_6 = [
        ["GPT-Neo 1.3B (fp32)", "10,75% → 11,35%", "21,51% → 21,61%", "11,90% → 11,65%", "— → 21,61%", "94,0% → 94,9%"],
        ["OPT 1.3B (fp32)", "12,13% → 12,29%", "22,10% → 21,96%", "11,78% → 12,24%", "— → 22,04%", "— → 90,7%"],
        ["GPT-Neo 2.7B (fp32)", "12,25% → 12,37%", "22,88% → 22,75%", "—", "23,31% → 23,27%", "91,0% → 90,8%"],
        ["OPT 2.7B (fp32)", "12,33% → 12,56%", "23,25% → 23,23%", "—", "— → 23,79%", "— → 88,5%"],
        ["GPT-J 6B (fp16)", "16,03% → 16,09%", "24,53% → 24,43%", "—", "25,55% → 25,56%", "84,4% → 83,3%"],
        ["OPT 6.7B (fp16)", "15,70% → 15,66%", "25,02% → 24,62%", "—", "— → 25,79%", "— → 83,7%"],
    ]
    add_styled_table(s, 0.55, 5.34, 12.23, 1.68, headers_6, rows_6,
                     col_widths=[2.35, 2.05, 2.05, 2.05, 2.08, 1.65], font_size=9.5)
    set_notes(s, (
        "Slide 6 (1,5 phút): Kết quả đối chiếu 6 mô hình gốc. Toàn bộ 16 ô so sánh đều nằm trong biên độ 0,01 đến 0,60 điểm %, "
        "MAD chỉ 0,18 điểm %. Đặc biệt GPT-J 6B Adaptive gần như trùng khít (25,55% vs 25,56%)."
    ))

    # =========================================================================
    # SLIDE 7: GIẢI THÍCH CHÊNH LỆCH NHẸ & BẰNG CHỨNG ĐỐI CHỨNG
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Giải thích chênh lệch nhẹ số liệu (0,01 – 0,60 điểm %)",
               "Vì sao số chạy lại lệch nhẹ so với paper gốc? Kiểm chứng bằng thí nghiệm đối chứng trên cùng 14.267 câu",
               "60% · ĐỐI CHỨNG", NAVY, 7)

    add_card(s, 0.55, 1.18, 6.0, 2.72, "1. Ví dụ mẫu BM25 có kèm đoạn văn (Bằng chứng mạnh nhất)", [
        ("Đối chứng trên đủ 14.267 câu (results/code_goc/): ",
         "Chạy run_model.py nguyên bản (không kèm đoạn cho 15 ví dụ mẫu): GPT-Neo 1.3B BM25 chỉ đạt 20,61% (kém paper -0,90pp, McNemar p < 10⁻⁷)."),
        ("Sau khi sửa kèm đoạn văn top-1: ",
         "GPT-Neo 1.3B BM25 đạt 21,61% — khớp sát paper 21,51% (+0,10pp). Điều này cho thấy khi viết paper, tác giả có kèm đoạn văn cho ví dụ mẫu."),
    ], accent=CORAL, size=12.0)

    add_card(s, 6.78, 1.18, 6.0, 2.72, "2. Độ chính xác số fp16 cho mô hình 6B / 6.7B", [
        ("Lý do kỹ thuật: ", "Paper chạy fp32 trên GPU V100 32 GB; nhóm dùng Kaggle T4 15 GB nên nạp fp16 cho GPT-J 6B và OPT 6.7B."),
        ("Mức ảnh hưởng đo được: ", "Sai số làm tròn dấu phẩy động rất nhỏ — GPT-J 6B chỉ lệch +0,06pp (Vanilla) và -0,10pp (BM25); OPT 6.7B lệch -0,04pp và -0,40pp."),
    ], accent=NAVY, size=12.0)

    add_card(s, 0.55, 4.10, 6.0, 2.72, "3. Ghép Adaptive theo id & Cố định ví dụ mẫu", [
        ("Sửa lỗi ghép 881 câu trùng văn bản: ", "Code gốc sort theo question nên có thể ghép sai cặp Vanilla/BM25; nhóm ghép chính xác theo id."),
        ("Cố định seed theo id & môi trường mới: ", "Khác phiên bản PyTorch/Transformers (2026 vs 2023) làm thay đổi nhẹ một vài câu sát biên quyết định (0,1–0,6pp ở Vanilla)."),
    ], accent=TEAL, size=12.0)

    # Bảng đối chứng nhỏ ở góc phải dưới
    add_textbox(s, 6.78, 4.10, 6.0, 0.34, "4. Bảng đối chứng code gốc nguyên bản vs bản sửa của nhóm (n = 14.267)",
                size=12.5, bold=True, color=NAVY)
    headers_7 = ["Mô hình · Cách", "Paper", "Code gốc (lỗi)", "Nhóm (đã sửa)", "McNemar p"]
    rows_7 = [
        ["GPT-Neo 1.3B · Vanilla", "10,75%", "11,17%", "11,35% (+0,18)", "0,259 (ns)"],
        ["GPT-Neo 1.3B · BM25", "21,51%", "20,61%", "21,61% (+1,00)", "9,3e-08 ***"],
        ["GPT-Neo 2.7B · Vanilla", "12,25%", "12,08%", "12,37% (+0,29)", "0,042 *"],
        ["GPT-Neo 2.7B · BM25", "22,88%", "22,95%", "22,75% (-0,20)", "0,286 (ns)"],
    ]
    add_styled_table(s, 6.78, 4.50, 6.0, 2.25, headers_7, rows_7,
                     col_widths=[1.85, 0.85, 1.15, 1.25, 0.90], font_size=9.8, highlight_rows={1})
    set_notes(s, (
        "Slide 7 (1,0 phút): Giải thích rõ vì sao có chênh lệch 0,01–0,60 điểm %. "
        "Nhấn mạnh bảng đối chứng ở góc phải: sửa lỗi ví dụ mẫu BM25 kéo GPT-Neo 1.3B từ 20,61% lên 21,61%, khớp sát paper 21,51%."
    ))

    # =========================================================================
    # SLIDE 8: KIỂM CHỨNG 4 KẾT LUẬN CỦA PAPER
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Kiểm chứng 4 kết luận của paper trên cả 9 mô hình",
               "Độ chính xác theo độ nổi tiếng tương đối (chuẩn hóa trong từng loại quan hệ, như Hình 8 của paper)",
               "60% · 4 KẾT LUẬN", NAVY, 8)

    add_picture_fit(s, FIGS / "acc_vs_pop.png", 0.55, 1.10, 7.65, 5.82)

    add_card(s, 8.38, 1.12, 4.40, 5.80, "Kết quả kiểm chứng (9/9 mô hình)", [
        ("✓ KL1 (Nổi tiếng → hay đúng): ", "ĐÚNG ở 9/9 mô hình. Pearson r = +0,09 → +0,23 (p < 10⁻²⁵); Q5 hơn Q1 tới +20pp."),
        ("✓ KL2 (Scaling ít cứu long-tail): ", "ĐÚNG ở cả 3 họ. Kiểm định DiD (Q5 vs Q1): GPT-Neo +16,0pp, OPT +11,7pp, Qwen2.5 +16,3pp (p < 10⁻³⁹)."),
        ("✓ KL3 (BM25 giúp Q1, hại Q5): ", "ĐÚNG. Vùng xanh (trái): Q1 tăng +16 đến +20pp. Vùng đỏ (phải): Q5 bị hại ở 4 model mạnh nhất (-1,9 đến -5,4pp, p < 0,01)."),
        ("✓ KL4 (Adaptive tối ưu & rẻ hơn): ", "ĐÚNG. Thắng BM25 ở 100/100 lần chia cho 5 model lớn (Wilcoxon p < 4×10⁻¹⁸), giảm tra cứu còn 78,6%–83,7%."),
    ], accent=TEAL, size=11.5, space_after=8)
    set_notes(s, (
        "Slide 8 (1,25 phút): Chỉ trực tiếp vào hình acc_vs_pop.png: bên trái (vùng xanh) đường BM25 luôn vượt xa Vanilla; "
        "bên phải (vùng đỏ ở GPT-J 6B, OPT 6.7B và Qwen2.5 3B) đường Vanilla vượt lên trên BM25 — chứng minh trực quan vì sao cần Adaptive."
    ))

    # =========================================================================
    # SLIDE 9: MỞ RỘNG A — QWEN2.5 (0.5B / 1.5B / 3B)
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Mở rộng A: Mô hình thế hệ mới Qwen2.5 (0.5B / 1.5B / 3B)",
               "Chạy cùng cấu hình 15-shot, cửa sổ 2.048 token (công bằng với GPT-Neo/OPT) trên đủ 14.267 câu PopQA",
               "30% · MỞ RỘNG A", CORAL, 9)

    headers_9 = ["Mô hình", "Cỡ", "Vanilla", "BM25", "Adaptive", "Tra cứu", "So sánh McNemar ghép cặp (n = 14.267)"]
    rows_9 = [
        ["GPT-Neo 1.3B / OPT 1.3B", "1.3B", "11,35 / 12,29%", "21,61 / 21,96%", "21,61 / 22,04%", "94,9 / 90,7%", "Baseline P0 (paper 2021–2022)"],
        ["GPT-Neo 2.7B / OPT 2.7B", "2.7B", "12,37 / 12,56%", "22,75 / 23,23%", "23,27 / 23,79%", "90,8 / 88,5%", "Baseline P1 (paper 2021–2022)"],
        ["GPT-J 6B / OPT 6.7B (fp16)", "6–6.7B", "16,09 / 15,66%", "24,43 / 24,62%", "25,56 / 25,79%", "83,3 / 83,7%", "Baseline P2 (mạnh nhất nhóm tái lập)"],
        ["Qwen2.5 0.5B (base, fp32)", "0.5B", "11,99%", "22,55%", "22,72%", "93,4%", "Nhỏ hơn 2,6× nhưng thắng GPT-Neo 1.3B (p < 10⁻³)"],
        ["Qwen2.5 1.5B (base, fp32)", "1.5B", "15,95%", "24,88%", "25,69%", "82,2%", "Nhỏ hơn 4×, ngang OPT 6.7B & thắng GPT-J BM25 (p=0,016)"],
        ["Qwen2.5 3B (base, fp32)", "3.0B", "18,35%", "26,40%", "27,65%", "78,6%", "Nhỏ hơn 2×, THẮNG ÁP ĐẢO 6B–6.7B (p < 10⁻²⁰)"],
    ]
    add_styled_table(s, 0.55, 1.15, 12.23, 3.05, headers_9, rows_9,
                     col_widths=[2.45, 0.80, 1.45, 1.45, 1.45, 1.15, 3.48], font_size=10.0, highlight_rows={5})

    add_kpi_card(s, 0.55, 4.42, 3.85, 1.15, "+2,26pp / +2,69pp",
                 "Qwen2.5 3B Vanilla vượt GPT-J 6B & OPT 6.7B\n(McNemar p = 9,7e-22 & 1,0e-29)", CORAL)
    add_kpi_card(s, 4.74, 4.42, 3.85, 1.15, "+1,98pp / +1,78pp",
                 "Qwen2.5 3B BM25 vượt GPT-J 6B & OPT 6.7B\n(giữ nguyên khi chấm chặt: p < 10⁻¹⁷)", TEAL)
    add_kpi_card(s, 8.93, 4.42, 3.85, 1.15, "27,65% · 78,6%",
                 "Adaptive cao nhất cả 9 model, tiệm cận GPT-NeoX 20B (28,14%) dù nhỏ hơn 6,7×", NAVY)

    add_rect(s, 0.55, 5.75, 12.23, 1.12, BLUE_BG, border_color=NAVY, border_pt=1.0, radius=True)
    add_bullets(s, 0.72, 5.84, 11.9, 0.95, [
        ("Kiểm chứng độ bền vững bằng chấm chặt (is_correct_strict): ", "Qwen2.5 3B vẫn thắng GPT-J 6B +2,26pp (Vanilla, p=8,2e-22) và +1,91pp (BM25, p=1,2e-25) → loại trừ giả thuyết 'chỉ thắng nhờ lỗi chấm chuỗi con'."),
        ("Quyết định dừng ở 3B (issue #31): ", "Qwen2.5 3B BM25 đã tốn ~21 giờ GPU Kaggle (3 phiên); đã đủ 3 mốc cỡ (0.5B → 1.5B → 3B) để kiểm định quy luật scaling."),
    ], size=11.5, space_after=4)
    set_notes(s, (
        "Slide 9 (1,25 phút): Trình bày kết quả Mở rộng A trên họ Qwen2.5. "
        "Nhấn mạnh hiệu quả tham số gấp 2–4 lần được chứng minh bằng kiểm định McNemar ghép cặp trên đủ 14.267 câu."
    ))

    # =========================================================================
    # SLIDE 10: TẠI SAO QWEN2.5 TỐT HƠN? — 3 CƠ CHẾ ĐỊNH LƯỢNG
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Tại sao Qwen2.5 vượt trội? Chứng minh 3 cơ chế định lượng",
               "Phân rã nguyên nhân từ dữ liệu từng câu hỏi (n = 14.267) kèm kiểm định thống kê & lưu ý khoa học",
               "30% · PHÂN TÍCH QWEN2.5", CORAL, 10)

    add_picture_fit(s, FIGS / "qwen_vs_paper_mechanisms.png", 0.55, 1.10, 12.23, 3.45)

    add_card(s, 0.55, 4.68, 3.92, 2.30, "Cơ chế 1 · Trí nhớ tham số dày hơn", [
        ("18T tokens (gấp 45–100× The Pile/OPT): ", "Đẩy ngưỡng nhớ xuống tầng trung bình Q4 (+3,89pp vs GPT-J, p = 2,1e-14)."),
        ("Vẫn bị chặn ở Long-Tail: ", "4.000 câu hiếm nhất chỉ 17,08%; DiD(Q5−Q1) = +16,34pp (p = 4,6e-68). Lưu ý: không loại trừ hoàn toàn rò rỉ dữ liệu vì Qwen2.5 ra sau PopQA."),
    ], accent=NAVY, size=10.6, space_after=4)

    add_card(s, 4.70, 4.68, 3.92, 2.30, "Cơ chế 2 · Đọc hiểu vượt trội nhưng vẫn sợ nhiễu", [
        ("Đọc đúng khi đoạn có đáp án (n=3.644): ", "Qwen2.5 3B đạt 86,99% vs GPT-J 82,33% (+4,67pp, p = 3,8e-20)."),
        ("Điểm yếu chưa giải quyết: ", "Khi đoạn BM25 sai trên câu Vanilla đang đúng, Qwen2.5 3B chỉ giữ đúng 46,55% (ngang OPT 6.7B: 47,57%) → vẫn bị hại -5,15pp ở Q5!"),
    ], accent=CORAL, size=10.6, space_after=4)

    add_card(s, 8.85, 4.68, 3.93, 2.30, "Cơ chế 3 · Calibration & Adaptive tốt nhất", [
        ("Tự biết mình biết gì (AUROC conf): ", "Qwen2.5 3B đạt 0,866 vs GPT-J 0,805 (+0,060, Bootstrap p < 0,002); ngay cả 0.5B (0,858) cũng vượt 6 model paper."),
        ("Adaptive tối ưu nhất: ", "Thắng BM25 ở 100/100 splits (Wilcoxon p = 3,8e-18), tăng +1,29pp và tiết kiệm 21,4% lượt tra cứu."),
    ], accent=TEAL, size=10.6, space_after=4)
    set_notes(s, (
        "Slide 10 (1,5 phút): Đây là slide trả lời câu hỏi 'TẠI SAO Qwen2.5 tốt hơn?' với 3 cơ chế có kiểm định. "
        "Nhấn mạnh phát hiện đối lập rất sắc: Qwen2.5 đọc đoạn đúng giỏi hơn hẳn (+4,67pp) nhưng gặp đoạn nhiễu vẫn dễ bị lừa ngang model cũ (~53% bị hỏng), "
        "và thành thật ghi nhận giới hạn data contamination."
    ))

    # =========================================================================
    # SLIDE 11: MỞ RỘNG C — ADAPTIVE THEO ĐỘ TỰ TIN (CONF)
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Mở rộng C: Cải tiến Adaptive bằng Độ tự tin (conf)",
               "Thay độ nổi tiếng s_pop (không có sẵn ngoài đời) bằng mean log-prob của chính câu trả lời Vanilla",
               "30% · MỞ RỘNG C", CORAL, 11)

    headers_11 = ["Mô hình", "Adaptive (s_pop)", "Adaptive (conf)", "Chênh (pp)", "Tra cứu: s_pop → conf", "AUROC (conf)", "AUROC (s_pop)"]
    rows_11 = [
        ["GPT-Neo 1.3B", "21,61%", "21,54%", "-0,07", "94,9% → 95,7%", "0,78", "0,53"],
        ["OPT 1.3B", "22,04%", "22,00%", "-0,04", "90,7% → 93,1%", "0,79", "0,56"],
        ["GPT-Neo 2.7B", "23,27%", "23,25%", "-0,02", "90,8% → 91,6%", "0,77", "0,56"],
        ["OPT 2.7B", "23,79%", "23,78%", "-0,01", "88,5% → 90,8%", "0,82", "0,62"],
        ["OPT 6.7B (fp16)", "25,79%", "25,65%", "-0,14", "83,7% → 87,0%", "0,83", "0,63"],
        ["GPT-J 6B (fp16)", "25,56%", "25,38%", "-0,18", "83,3% → 89,5%", "0,81", "0,63"],
        ["Qwen2.5 0.5B", "22,72%", "22,65%", "-0,07", "93,4% → 92,0%", "0,86", "0,53"],
        ["Qwen2.5 1.5B", "25,69%", "25,67%", "-0,02", "82,2% → 87,5%", "0,83", "0,58"],
        ["Qwen2.5 3B", "27,65%", "27,53%", "-0,12", "78,6% → 84,9%", "0,87", "0,63"],
    ]
    add_styled_table(s, 0.55, 1.15, 7.65, 5.75, headers_11, rows_11,
                     col_widths=[1.65, 1.05, 1.05, 0.85, 1.35, 0.85, 0.85], font_size=9.5, highlight_rows={8})

    add_card(s, 8.42, 1.15, 4.36, 2.72, "Kết quả cải tiến: Thay thế được!", [
        ("Độ chính xác tương đương: ", "Chỉ thấp hơn 0,01–0,18pp (nằm gọn trong độ lệch chuẩn 0,54–0,66pp), không cần tra Wikipedia pageview trước."),
        ("Đánh đổi chi phí: ", "Phải tra cứu nhiều hơn ở 8/9 mô hình (+0,8 đến +6,3pp; chỉ Qwen2.5 0.5B giảm nhẹ 93,4% → 92,0%)."),
    ], accent=TEAL, size=11.5)

    add_card(s, 8.42, 4.08, 4.36, 2.82, "Nghịch lý thú vị: Vì sao AUROC cao hơn mà không thắng?", [
        ("conf dự đoán 'model đúng/sai' cực tốt: ", "AUROC 0,77–0,87 (vượt xa s_pop chỉ 0,53–0,63)."),
        ("Nhưng Adaptive cần dự đoán 'BM25 giúp hay hại': ",
         "s_pop tách thẳng vùng bị hại ở Q5 (-1,9 đến -5,4pp) cho 4 model mạnh; còn conf chỉ thấy vùng bị hại ở duy nhất Qwen2.5 3B (-3,0pp)."),
        ("Kết quả trung thực: ", "Báo cáo đúng kết quả 'ngang điểm, chưa thắng chi phí' — hướng mở là kết hợp cả conf + tín hiệu truy xuất."),
    ], accent=AMBER, size=11.2, space_after=4)
    set_notes(s, (
        "Slide 11 (1,25 phút): Giải thích Hướng C — tại sao nhóm chọn độ tự tin conf (vì thực tế không có s_pop và tận dụng được kết quả vanilla sẵn có). "
        "Giải thích nghịch lý AUROC rất ấn tượng với giảng viên: biết mình đúng/sai khác với biết retrieval có làm hại mình không."
    ))

    # =========================================================================
    # SLIDE 12: PHÂN TÍCH 10% (1 & 2) — CHẤM CHẶT & 16 LOẠI QUAN HỆ
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Phân tích sâu (10%) · Chấm chặt hơn & 16 loại quan hệ",
               "Kiểm tra độ tin cậy của cách chấm chuỗi con (issue #35) và cấu trúc độ khó theo quan hệ (issue #36)",
               "10% · PHÂN TÍCH 1–2", TEAL, 12)

    add_picture_fit(s, FIGS / "acc_corr_by_relation.png", 0.55, 1.15, 7.25, 5.75)

    add_card(s, 8.00, 1.15, 4.78, 2.75, "1. Chấm chặt hơn (is_correct_strict)", [
        ("Vấn đề cách chấm gốc: ", "Chỉ cần đáp án là chuỗi con → 'Russia' và 'Australia' bị chấm ĐÚNG cho đáp án 'US' vì chứa 'us'!"),
        ("Mức thổi điểm đo được: ", "Chấm chặt (chuẩn hóa, khớp cả từ đứng đầu) chỉ hạ 0,23–0,73 điểm % (1,5–4% điểm cũ) trên toàn bộ 291.340 dòng."),
        ("Kết luận giữ nguyên: ", "Thứ hạng 9 mô hình không đổi; Adaptive thậm chí thắng/bằng cả hai cách trên 9/9 mô hình."),
    ], accent=NAVY, size=11.3, space_after=4)

    add_card(s, 8.00, 4.10, 4.78, 2.80, "2. Độ chính xác theo 16 loại quan hệ", [
        ("Độ khó chênh 27,6 lần: ", "Vanilla dễ nhất ở capital (61,7%), sport (57,4%), country (55,1%); khó nhất ở producer (2,2%), director (2,7%), screenwriter (3,1%)."),
        ("BM25 làm phẳng tương quan độ nổi tiếng: ", "Hệ số tương quan giảm ở 16/16 loại quan hệ (director: 0,29 → 0,01)."),
        ("Ngoại lệ duy nhất: ", "occupation là quan hệ duy nhất BM25 làm giảm điểm (-2,1pp); place of birth tăng mạnh nhất (+34,8pp)."),
    ], accent=TEAL, size=11.3, space_after=4)
    set_notes(s, (
        "Slide 12 (0,75 phút): Tóm gọn 2 phân tích đầu của phần 10%. Nhấn mạnh ví dụ vui nhưng đắt giá: "
        "'Russia' bị chấm đúng cho nước Mỹ vì chữ 'us' nằm trong R-us-sia, nhưng chấm chặt hơn không đổi bất kỳ kết luận nào."
    ))

    # =========================================================================
    # SLIDE 13: PHÂN TÍCH 10% (3) — RETRIEVAL GÂY HẠI & ĐOẠN DÙNG CHUNG
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Phân tích sâu (10%) · Vì sao Retrieval gây hại?",
               "Bảng 2×2 BM25 (issue #29) và phát hiện 'Đoạn dùng chung' phủ 1/3 bộ PopQA (issue #37)",
               "10% · PHÂN TÍCH 3", TEAL, 13)

    add_kpi_card(s, 0.55, 1.15, 3.85, 1.02, "33,6% (4.787 câu)",
                 "Nhận 1 trong 64 'đoạn dùng chung' (top-1 cho ≥ 5 chủ thể khác nhau)", CORAL)
    add_kpi_card(s, 4.74, 1.15, 3.85, 1.02, "1,0% vs 38,0%",
                 "recall@1 trên nhóm đoạn dùng chung so với 2/3 câu còn lại (chung: 25,54%)", NAVY)
    add_kpi_card(s, 8.93, 1.15, 3.85, 1.02, "35,6% – 54,8%",
                 "Tỉ lệ câu bị hại tập trung ở nhóm Q5 nổi tiếng nhất (vì có nhiều thứ để mất)", AMBER)

    add_card(s, 0.55, 2.35, 6.0, 4.55, "Phát hiện chính: BM25 sụp đổ ở 1/3 câu hỏi vì đoạn dùng chung", [
        ("Cơ chế hỏng của BM25: ", "Khi không tìm thấy tên chủ thể hiếm, BM25 khớp vào từ khóa loại quan hệ (như 'producer', 'screenwriter') → trả về cùng 1 đoạn chung chung cho hàng trăm câu hỏi khác nhau."),
        ("Ví dụ đoạn dùng chung: ", "Đoạn về Sophie Fante ('television producer') được dùng cho 1.156 câu producer (984 chủ thể khác nhau); phủ tới 79% producer, 52% screenwriter, 50% author."),
        ("Tác động trái ngược: ", "Trên 1/3 câu có đoạn dùng chung, BM25 làm GIẢM độ chính xác ở cả 9/9 mô hình; trên 2/3 câu còn lại, BM25 tăng vọt +13,8 đến +16,6 điểm %!"),
        ("Ô 'hại' trong bảng 2×2: ", "Tăng theo độ mạnh model (2,3% ở GPT-Neo 1.3B → 4,5% ở Qwen2.5 3B), chiếm 20–27% số câu Vanilla vốn làm đúng."),
    ], accent=CORAL, size=11.5, space_after=5)

    add_card(s, 6.78, 2.35, 6.0, 4.55, "Ví dụ thật bị hại ở cả 9/9 mô hình (Vanilla ĐÚNG → BM25 SAI)", [
        ("1. Thực thể chứa tên chủ thể — 'What is the capital of Samoa?': ",
         "Đáp án: Apia. Cả 9/9 model Vanilla trả lời đúng Apia. BM25 trả đoạn về 'American Samoa Community College... Pago Pago, the capital of the territory' → cả 9/9 model chép theo Pago Pago!"),
        ("2. Thực thể trùng tên — 'What sport does Albert Cox play?': ",
         "Chủ thể là cầu thủ bóng đá (9/9 Vanilla đúng football); BM25 lấy đoạn về thủ môn khúc côn cầu cùng tên → 9/9 model chuyển sang ice hockey."),
        ("3. Trùng 1 từ trong tên — 'In what city was Léon-Lévy Brunswick born?': ",
         "9/9 Vanilla đúng Paris; BM25 khớp chữ 'Brunswick' vào tỉnh New Brunswick (Canada) → cả 9/9 model bỏ Paris."),
    ], accent=NAVY, size=11.5, space_after=6)
    set_notes(s, (
        "Slide 13 (1,0 phút): Đây là phát hiện độc đáo nhất của nhóm trong phần 10%. "
        "Kể ví dụ Samoa → Pago Pago để cả lớp thấy rõ vì sao retrieval không hại ngẫu nhiên mà do đoạn văn trùng tên / dùng chung."
    ))

    # =========================================================================
    # SLIDE 14: PHÂN TÍCH 10% (4) — ĐOẠN CÓ ĐÁP ÁN MÀ VẪN SAI
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Phân tích sâu (10%) · Đoạn BM25 có đáp án mà vẫn sai",
               "Trong 3.644 câu đoạn top-1 chứa đáp án, mô hình vẫn bị chấm sai 13,0%–24,5%: lỗi đọc hay lỗi đo? (issue #38)",
               "10% · PHÂN TÍCH 4", TEAL, 14)

    add_kpi_card(s, 0.55, 1.15, 3.85, 1.02, "75,5% – 87,0%",
                 "Tỉ lệ đọc đúng thực tế khi đoạn có đáp án (cao nhất: Qwen2.5 3B 87,0%)", TEAL)
    add_kpi_card(s, 4.74, 1.15, 3.85, 1.02, "75% (15 / 20 ca)",
                 "Đọc tay mẫu GPT-J: phần lớn 'sai' là lỗi đo hoặc đáp án chuẩn hẹp", CORAL)
    add_kpi_card(s, 8.93, 1.15, 3.85, 1.02, "0 ca bị cắt",
                 "Đã kiểm chứng loại trừ giả thuyết đáp án bị cắt do giới hạn 2.048 token", NAVY)

    headers_14 = ["Nhóm nguyên nhân (Đọc tay 20 ca GPT-J)", "Số ca", "Ví dụ thực tế từ PopQA"]
    rows_14 = [
        ["Đúng/gần đúng về nghĩa nhưng chấm sai", "7 (35%)", "Hỏi thể loại City Slickers: trả lời 'comedy' bị chấm sai so với 'comedy film'; 'Sooraj Barjatya' vs 'Sooraj R. Barjatya'."],
        ["Có đáp án giả (khớp chuỗi con vô tình)", "5 (25%)", "'pol' (politician) nằm trong 'Marco Polo'; 'legal thriller' xuất hiện trong câu nói phim nằm NGOÀI thể loại đó."],
        ["Đáp án chuẩn quá hẹp", "3 (15%)", "Đoạn ghi 'produced by William LeBaron' nhưng đáp án chuẩn chỉ nhận hãng phim '20th Century Studios'."],
        ["Mô hình chọn nhầm ứng viên trong đoạn", "3 (15%)", "Hỏi cha Công chúa Mary: đoạn có cả cha (King George III) lẫn chồng → model chép nhầm tên chồng."],
        ["Mô hình bỏ qua đoạn văn", "2 (10%)", "Đoạn viết thẳng 'Orotina is the capital of Orotina Canton' nhưng model vẫn trả lời 'San José'."],
    ]
    add_styled_table(s, 0.55, 2.38, 12.23, 3.25, headers_14, rows_14,
                     col_widths=[3.45, 1.15, 7.63], font_size=10.8, highlight_rows={0, 1})

    add_rect(s, 0.55, 5.82, 12.23, 1.05, TEAL_BG, border_color=TEAL, border_pt=1.0, radius=True)
    add_bullets(s, 0.72, 5.92, 11.9, 0.88, [
        ("Kết luận tự động trên toàn bộ 9 mô hình: ", "Ít nhất 29,3%–45,6% số câu 'sai' thuộc nhóm đáp án giả hoặc gần đúng về nghĩa (cận dưới). Quan hệ religion (59–68%) và genre (25–50%) có tỉ lệ sai cao nhất vì có nhiều cách diễn đạt đúng khác nhau."),
        ("Hàm ý: ", "Năng lực đọc hiểu thực sự của LLM khi được cấp đúng ngữ cảnh cao hơn đáng kể so với con số đo bằng exact substring match."),
    ], size=11.4, space_after=4)
    set_notes(s, (
        "Slide 14 (0,75 phút): Cho thấy khi có đáp án trong đoạn văn, chỉ khoảng 25% số ca sai là mô hình thực sự đọc sai; "
        "75% còn lại là do thước đo so chuỗi quá cứng nhắc hoặc đáp án chuẩn Wikidata quá hẹp."
    ))

    # =========================================================================
    # SLIDE 15: TỔNG KẾT, HẠN CHẾ & KHẢ NĂNG TÁI LẬP
    # =========================================================================
    s = prs.slides.add_slide(blank)
    add_header(s, "Tổng kết, Hạn chế & Khả năng tái lập (Reproducibility)",
               "Bốn thông điệp chính từ đồ án và hướng dẫn kiểm chứng nhanh toàn bộ thực nghiệm",
               "TỔNG KẾT", NAVY, 15)

    add_card(s, 0.55, 1.18, 6.25, 4.15, "4 thông điệp chính rút ra", [
        ("1. Tái lập thành công sát paper: ", "6 mô hình gốc trên đủ 14.267 câu PopQA khớp paper với MAD = 0,18 điểm %; cả 4 kết luận của Mallen et al. (ACL 2023) đều được xác nhận."),
        ("2. Qwen2.5 vượt trội 2–4× cỡ nhưng không phá vỡ quy luật paper: ", "Mạnh hơn nhờ trí nhớ tham số dày hơn (18T tokens), đọc hiểu tốt hơn (+4,7pp) và calibration cao hơn (AUROC 0,866) — nhưng vẫn yếu ở long-tail (17,1%) và vẫn bị đoạn nhiễu đánh lừa (~53%)."),
        ("3. Adaptive theo độ tự tin (conf) khả thi thực tế: ", "Thay được s_pop với độ chính xác tương đương (-0,01 đến -0,18pp), dù phải tra cứu nhiều hơn một chút."),
        ("4. Chất lượng truy xuất & thước đo là nút thắt lớn: ", "64 đoạn dùng chung phủ 33,6% PopQA; phần lớn lỗi khi đã có đáp án đến từ quy tắc so chuỗi."),
    ], accent=NAVY, size=11.5, space_after=6)

    add_card(s, 7.05, 1.18, 5.73, 2.50, "Hạn chế & Hướng phát triển", [
        ("Phạm vi: ", "Chưa chạy được GPT-3, Contriever (mất file), model ≥13B và EntityQuestions; Qwen2.5 có nguy cơ rò rỉ dữ liệu thời gian (2024 vs 2023)."),
        ("Hướng mở: ", "Kết hợp độ tự tin conf với điểm tương đồng truy xuất (retrieval score / reranker) để vừa bỏ được s_pop vừa lọc được 64 đoạn dùng chung."),
    ], accent=AMBER, size=11.5, space_after=5)

    add_card(s, 7.05, 3.85, 5.73, 1.48, "Tái lập & Kiểm chứng trên GitHub (1 lệnh)", [
        ("Repo: ", "github.com/vanductan-NLT/CS221 (README 3 mức + requirements.txt)."),
        ("Kiểm chứng CPU (~30s): ", "pip install -r requirements.txt  &&  python scripts/run_all_tests.py  (9/9 PASS)."),
    ], accent=TEAL, bg=TEAL_BG, size=11.4, space_after=4)

    add_rect(s, 0.55, 5.58, 12.23, 1.30, NAVY, radius=True)
    add_textbox(s, 0.85, 5.76, 11.6, 0.52,
                "Cảm ơn Cô và các bạn đã lắng nghe!",
                size=24, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(s, 0.85, 6.32, 11.6, 0.40,
                "Nhóm sẵn sàng trao đổi và trả lời câu hỏi Q&A.",
                size=13.5, color=RGBColor(0x93, 0xC5, 0xFD), align=PP_ALIGN.CENTER)
    set_notes(s, (
        "Slide 15 (0,5 phút): Chốt lại 4 thông điệp chính, nêu hạn chế trung thực và mời cô cùng lớp đặt câu hỏi Q&A."
    ))

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print(f"Đã tạo thành công {OUT_PPTX} ({len(prs.slides)} slides, {OUT_PPTX.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
