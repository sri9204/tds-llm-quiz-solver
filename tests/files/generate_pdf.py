import os
from reportlab.platypus import SimpleDocTemplate, Table, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# Output PDF path
output_folder = "tests/files"
output_path = os.path.join(output_folder, "sample_q834.pdf")

# Ensure folder exists
os.makedirs(output_folder, exist_ok=True)

# Create the PDF
doc = SimpleDocTemplate(output_path, pagesize=letter)
story = []

# --- PAGE 1 ---
story.append(Table([["This is page 1 (blank as per quiz format)."]]))
story.append(Spacer(1, 600))  # Force page break with a big spacer

# --- PAGE 2: TABLE ---
data = [
    ["id", "value"],
    ["1", "10"],
    ["2", "20"],
    ["3", "30"]
]

table = Table(data, style=[
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
])

story.append(table)

# Build PDF
doc.build(story)

print("PDF created successfully:", output_path)
