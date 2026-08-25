from pathlib import Path
from lxml import html
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
HTML_FILE = ROOT / "Assembly_Copilot_Technical_Documentation.html"
OUT_FILE = ROOT / "Assembly_Copilot_Technical_Documentation.pdf"
W, H = 1240, 1754
M = 105
CONTENT_W = W - 2 * M
BLUE = (26, 54, 104)
LIGHT = (238, 243, 250)
GRID = (122, 135, 156)
TEXT = (35, 35, 35)

SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def open_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        base = Image.new("RGB", im.size, "white")
        base.paste(im, mask=im.getchannel("A"))
        return base
    return im.convert("RGB")


BODY = font(SERIF, 21)
BODY_BOLD = font(SERIF_BOLD, 21)
SMALL = font(SERIF, 17)
SMALL_BOLD = font(SERIF_BOLD, 17)
H2 = font(SERIF_BOLD, 29)
CHAPTER = font(SERIF_BOLD, 38)
TITLE = font(SERIF_BOLD, 41)
TOC = font(SERIF, 24)


def wrap(draw, text, fnt, width):
    lines = []
    for para in text.replace("\xa0", " ").splitlines() or [""]:
        words = para.split()
        if not words:
            lines.append("")
            continue
        cur = words[0]
        for word in words[1:]:
            trial = cur + " " + word
            if draw.textlength(trial, font=fnt) <= width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


class Document:
    def __init__(self):
        self.pages = []
        self.page = None
        self.draw = None
        self.y = M
        self.page_no = 0

    def new_page(self, footer=True):
        if self.page is not None:
            if footer:
                self.footer()
            self.pages.append(self.page.convert("RGB"))
        self.page = Image.new("RGB", (W, H), "white")
        self.draw = ImageDraw.Draw(self.page)
        self.y = M
        self.page_no += 1

    def footer(self):
        y = H - 65
        self.draw.line((M, y - 15, W - M, y - 15), fill=BLUE, width=2)
        self.draw.text((M, y), "Assembly Copilot Technical Documentation",
                       font=font(SANS, 14), fill=(80, 80, 80))
        label = str(self.page_no)
        tw = self.draw.textlength(label, font=font(SANS, 14))
        self.draw.text((W - M - tw, y), label, font=font(SANS, 14),
                       fill=(80, 80, 80))

    def ensure(self, height):
        if self.y + height > H - 100:
            self.new_page()

    def paragraph(self, text, fnt=BODY, fill=TEXT, gap=12, width=None):
        text = " ".join(text.split())
        width = width or CONTENT_W
        lines = wrap(self.draw, text, fnt, width)
        line_h = int(fnt.size * 1.36)
        self.ensure(len(lines) * line_h + gap)
        for line in lines:
            self.draw.text((M, self.y), line, font=fnt, fill=fill)
            self.y += line_h
        self.y += gap

    def heading(self, text):
        line_h = int(H2.size * 1.25)
        self.ensure(line_h + 160)
        self.draw.rectangle((M, self.y + 3, M + 7, self.y + line_h + 4), fill=BLUE)
        self.draw.text((M + 18, self.y), " ".join(text.split()),
                       font=H2, fill=BLUE)
        self.y += line_h + 14

    def chapter_title(self, text):
        self.ensure(70)
        self.draw.rectangle((M, self.y, W - M, self.y + 66), fill=BLUE)
        label = " ".join(text.split())
        title_font = CHAPTER
        while self.draw.textlength(label, font=title_font) > CONTENT_W - 40:
            title_font = font(SERIF_BOLD, title_font.size - 1)
        yy = self.y + max(8, (66 - title_font.size) // 2 - 2)
        self.draw.text((M + 20, yy), label, font=title_font, fill="white")
        self.y += 88

    def list_items(self, items, ordered=False):
        fnt = BODY
        line_h = int(fnt.size * 1.34)
        for i, item in enumerate(items, 1):
            lines = wrap(self.draw, " ".join(item.split()), fnt, CONTENT_W - 48)
            self.ensure(len(lines) * line_h + 8)
            marker = f"{i}." if ordered else "•"
            self.draw.text((M + 2, self.y), marker, font=BODY_BOLD, fill=BLUE)
            for j, line in enumerate(lines):
                self.draw.text((M + 40, self.y), line, font=fnt, fill=TEXT)
                self.y += line_h
            self.y += 5

    def callout(self, text):
        fnt = font(SANS_BOLD, 18)
        lines = wrap(self.draw, " ".join(text.split()), fnt, CONTENT_W - 38)
        line_h = int(fnt.size * 1.35)
        box_h = len(lines) * line_h + 28
        self.ensure(box_h + 18)
        self.draw.rounded_rectangle((M, self.y, W - M, self.y + box_h),
                                    radius=3, fill=LIGHT, outline=BLUE, width=2)
        yy = self.y + 14
        for line in lines:
            tw = self.draw.textlength(line, font=fnt)
            self.draw.text(((W - tw) / 2, yy), line, font=fnt, fill=BLUE)
            yy += line_h
        self.y += box_h + 18

    def image(self, path, tall=False, arch=False, full_arch=False):
        try:
            im = open_rgb(ROOT / path)
        except Exception:
            return
        max_w = CONTENT_W - 20
        max_h = 1150 if full_arch else (480 if arch else (460 if tall else 430))
        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.ensure(im.height + 30)
        x = (W - im.width) // 2
        self.page.paste(im, (x, self.y))
        self.y += im.height + 15

    def caption(self, text):
        fnt = font(SERIF, 16)
        lines = wrap(self.draw, " ".join(text.split()), fnt, CONTENT_W - 80)
        line_h = 21
        self.ensure(len(lines) * line_h + 12)
        for line in lines:
            tw = self.draw.textlength(line, font=fnt)
            self.draw.text(((W - tw) / 2, self.y), line, font=fnt, fill=(70, 70, 70))
            self.y += line_h
        self.y += 8

    def table(self, table):
        rows = table.xpath("./tr")
        if not rows:
            return
        self.ensure(130 + len(rows) * 42)
        n = len(rows[0].xpath("./th|./td"))
        if n == 2:
            proportions = [0.34, 0.66]
        elif n == 3:
            proportions = [0.12, 0.43, 0.45]
        elif n == 4:
            proportions = [0.60, 0.13, 0.13, 0.14]
        else:
            proportions = [0.25, 0.12, 0.12, 0.14, 0.37]
        widths = [int(CONTENT_W * p) for p in proportions]
        compact = "compact" in (table.get("class") or "")
        fnt = font(SANS, 13 if compact else 15)
        bold = font(SANS_BOLD, 13 if compact else 15)
        line_h = 17 if compact else 20
        for ri, row in enumerate(rows):
            cells = row.xpath("./th|./td")
            values = [" ".join(c.text_content().split()) for c in cells]
            wrapped = [wrap(self.draw, v, bold if ri == 0 else fnt,
                            widths[i] - 16) for i, v in enumerate(values)]
            row_h = max([len(x) for x in wrapped] + [1]) * line_h + 16
            self.ensure(row_h + 4)
            x = M
            is_header = ri == 0
            is_best = any(k in " ".join(values).lower()
                          for k in ["best", "final online", "robust_v1"])
            fill = BLUE if is_header else ((200, 220, 164) if is_best
                                           else ((243, 246, 250) if ri % 2 == 0
                                                 else "white"))
            for ci, cell_lines in enumerate(wrapped):
                self.draw.rectangle((x, self.y, x + widths[ci],
                                     self.y + row_h), fill=fill,
                                    outline=GRID, width=1)
                yy = self.y + 8
                cfnt = bold if is_header or is_best else fnt
                cfill = "white" if is_header else TEXT
                for line in cell_lines:
                    self.draw.text((x + 8, yy), line, font=cfnt, fill=cfill)
                    yy += line_h
                x += widths[ci]
            self.y += row_h
        self.y += 14

    def cover(self):
        self.new_page(footer=False)
        logo = open_rgb(ROOT / "Images/LM-logo-Black.png")
        logo.thumbnail((600, 190), Image.Resampling.LANCZOS)
        self.page.paste(logo, ((W - logo.width) // 2, 150))
        org = font(SERIF, 24)
        y = 600
        for line in ["Center for Innovation and Security Solutions",
                     "Technical Documentation"]:
            tw = self.draw.textlength(line, font=org)
            self.draw.text(((W - tw) / 2, y), line, font=org, fill=TEXT)
            y += 36
        y += 18
        self.draw.line((0, y, W, y), fill=BLUE, width=6)
        y += 22
        title_lines = wrap(self.draw,
            "Assembly Copilot: Procedure Step Recognition from Research to Operational Demonstration",
            TITLE, CONTENT_W - 20)
        for line in title_lines:
            tw = self.draw.textlength(line, font=TITLE)
            self.draw.text(((W - tw) / 2, y), line, font=TITLE, fill=TEXT)
            y += 53
        y += 18
        self.draw.line((0, y, W, y), fill=BLUE, width=6)
        by = font(SERIF, 23)
        lines = ["AIOPS Team"]
        y += 60
        for line in lines:
            tw = self.draw.textlength(line, font=by)
            self.draw.text(((W - tw) / 2, y), line, font=by, fill=TEXT)
            y += 35
        self.draw.text((W // 2 - 35, H - 190), "2026", font=by, fill=TEXT)

    def toc_page(self, toc):
        self.new_page()
        self.draw.text((M, self.y), "Contents", font=font(SERIF_BOLD, 40), fill=TEXT)
        self.y += 75
        for p in toc.xpath("./p"):
            txt = " ".join(p.text_content().split())
            fnt = TOC if "sub" not in (p.get("class") or "") else font(SERIF, 20)
            x = M if "sub" not in (p.get("class") or "") else M + 48
            self.ensure(35)
            self.draw.text((x, self.y), txt, font=fnt, fill=TEXT)
            self.y += 35 if fnt == TOC else 28

    def render_chapter(self, node):
        self.new_page()
        for child in node:
            tag = child.tag if isinstance(child.tag, str) else ""
            if tag == "div" and "chapter-title" in (child.get("class") or ""):
                self.chapter_title(child.text_content())
            elif tag == "div" and "pagebreak" in (child.get("class") or ""):
                self.new_page()
                title = child.get("data-title")
                if title:
                    self.chapter_title(title)
            elif tag == "h2" or tag == "h3":
                self.heading(child.text_content())
            elif tag == "p":
                self.paragraph(child.text_content())
            elif tag in ("ol", "ul"):
                items = [x.text_content() for x in child.xpath("./li")]
                self.list_items(items, ordered=(tag == "ol"))
            elif tag == "table":
                self.table(child)
            elif tag == "figure":
                img = child.xpath("./img")
                if img:
                    classes = child.get("class") or ""
                    self.image(img[0].get("src"), tall=("tall" in classes),
                               arch=("arch" in classes),
                               full_arch=("full-arch" in classes))
            elif tag == "div" and "caption" in (child.get("class") or ""):
                self.caption(child.text_content())
            elif tag == "div" and "callout" in (child.get("class") or ""):
                self.callout(child.text_content())
            elif tag == "div" and "refs" in (child.get("class") or ""):
                for p in child.xpath("./p"):
                    self.paragraph(p.text_content(), SMALL, gap=7)

    def save(self):
        if self.page is not None:
            self.footer()
            self.pages.append(self.page.convert("RGB"))
        self.pages[0].save(OUT_FILE, "PDF", resolution=150.0,
                           save_all=True, append_images=self.pages[1:])


root = html.parse(str(HTML_FILE)).getroot()
doc = Document()
doc.cover()
doc.toc_page(root.xpath("//div[@class='toc']")[0])
for chapter in root.xpath("//div[@class='chapter']"):
    doc.render_chapter(chapter)
doc.save()
print(f"wrote {OUT_FILE} ({len(doc.pages)} pages)")
