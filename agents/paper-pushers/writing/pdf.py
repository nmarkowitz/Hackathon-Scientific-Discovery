import base64
import re
import tempfile
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from hackathon_science.models import Paper


_BASE64_IMG_RE = re.compile(r'!\[([^\]]*)\]\(data:image/(png|jpeg);base64,([^)]+)\)')
_MD_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
_MD_HEADING_RE = re.compile(r'^#{1,3}\s+(.+)$')


class _PaperPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)


def _strip_base64_images(text: str) -> tuple[str, list[tuple[str, bytes]]]:
    images = []
    def replacer(m):
        alt = m.group(1) or "Figure"
        images.append((alt, base64.b64decode(m.group(3))))
        return f"[{alt}]"
    return _BASE64_IMG_RE.sub(replacer, text), images


def _write_section(pdf: _PaperPDF, heading: str, text: str) -> None:
    if not text.strip():
        return

    page_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(page_w, 8, heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(2)

    cleaned, images = _strip_base64_images(text)

    for line in cleaned.splitlines():
        pdf.set_x(pdf.l_margin)
        m = _MD_HEADING_RE.match(line)
        if m:
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(page_w, 6, m.group(1), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 10)
            continue

        plain = _MD_BOLD_RE.sub(r'\1', line).strip()
        pdf.set_font("Helvetica", "", 10)
        if plain:
            pdf.multi_cell(page_w, 5, plain, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.ln(3)

    for alt, img_bytes in images:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        try:
            pdf.set_x(pdf.l_margin)
            pdf.image(tmp_path, w=min(page_w, 120))
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "I", 8)
            pdf.multi_cell(page_w, 4, alt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
        except Exception:
            pass
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    pdf.ln(4)


def save_paper_as_pdf(paper: Paper, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_w = 210 - 40  # A4 minus margins

    pdf = _PaperPDF()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(page_w, 10, paper.title, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    if paper.author or paper.date:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "I", 10)
        meta = " | ".join(x for x in [paper.author, paper.date] if x)
        pdf.cell(page_w, 6, meta, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if paper.tags:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(page_w, 5, "Tags: " + ", ".join(paper.tags), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(6)
    pdf.set_draw_color(180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)

    _write_section(pdf, "Introduction", paper.introduction)
    _write_section(pdf, "Methods", paper.methods)
    _write_section(pdf, "Results", paper.results)
    if paper.references:
        _write_section(pdf, "References", paper.references)
    if paper.appendix:
        _write_section(pdf, "Appendix", paper.appendix)

    pdf.output(str(output_path))
    return output_path
