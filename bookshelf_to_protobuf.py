"""
Convert Bookshelf → Protobuf (.plc) using TILOS MacroPlacement FormatTranslators.

Prerequisites:
    git clone https://github.com/TILOS-AI-Institute/MacroPlacement.git
    pip install openroad  # or build OpenDB from source

Then set TRANSLATORS_SRC_PATH below.
"""

import sys

TRANSLATORS_SRC_PATH = "./MacroPlacement/CodeElements/FormatTranslators/src"
sys.path.append(TRANSLATORS_SRC_PATH)

from FormatTranslators import BookShelf2ProBufFormat

DESIGN = "ibm01"
BOOKSHELF_DIR = f"./bookshelf/{DESIGN}"
OUTPUT_PLC = f"{DESIGN}.plc"

BookShelf2ProBufFormat(BOOKSHELF_DIR, DESIGN, OUTPUT_PLC)
print(f"Protobuf netlist written to: {OUTPUT_PLC}")