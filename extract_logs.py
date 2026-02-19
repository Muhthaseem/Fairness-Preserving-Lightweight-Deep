"""
Extract training logs from deepfake_training.html and save to outputs/logs/
"""
from html.parser import HTMLParser
import os
import re

HTML_PATH = r"D:\M3 Projects\DeepFake_Research\deepfake_training.html"
LOGS_DIR = r"D:\M3 Projects\DeepFake_Research\outputs\logs"
os.makedirs(LOGS_DIR, exist_ok=True)


class OutputExtractor(HTMLParser):
    """Extract text from notebook output cells."""
    def __init__(self):
        super().__init__()
        self.in_output = False
        self.in_pre = False
        self.outputs = []
        self.current = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")
        # Detect output areas
        if "output" in cls and "cell" not in cls:
            self.in_output = True
            self.depth += 1
        if tag == "pre" and self.in_output:
            self.in_pre = True

    def handle_endtag(self, tag):
        if tag == "pre" and self.in_pre:
            self.in_pre = False
        if self.in_output and tag == "div":
            self.depth -= 1
            if self.depth <= 0:
                if self.current:
                    self.outputs.append("".join(self.current))
                    self.current = []
                self.in_output = False
                self.depth = 0

    def handle_data(self, data):
        if self.in_output:
            self.current.append(data)


with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

parser = OutputExtractor()
parser.feed(html)

# Flush any remaining
if parser.current:
    parser.outputs.append("".join(parser.current))

# Combine all outputs into one big training log
all_logs = []
for i, output in enumerate(parser.outputs):
    text = output.strip()
    if text:
        all_logs.append(text)

full_log = "\n\n" + ("=" * 70 + "\n").join(
    [f"[Cell Output {i+1}]\n{log}\n" for i, log in enumerate(all_logs)]
)

# Save the full training log
log_path = os.path.join(LOGS_DIR, "training_log.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(full_log)

print(f"✅ Saved {len(all_logs)} cell outputs to: {log_path}")
print(f"   Total size: {len(full_log):,} characters")

# Also copy the HTML as a complete record
import shutil
html_copy = os.path.join(LOGS_DIR, "deepfake_training_full.html")
shutil.copy2(HTML_PATH, html_copy)
print(f"✅ Copied full HTML to: {html_copy}")
