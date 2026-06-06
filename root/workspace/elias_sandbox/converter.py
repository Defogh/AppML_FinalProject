import json

input_file = "Testing3.ipynb"
output_file = "notebook_export.txt"

with open(input_file, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(output_file, "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb.get("cells", []), start=1):
        cell_type = cell.get("cell_type", "unknown")
        out.write(f"\n{'='*60}\n")
        out.write(f"CELL {i} [{cell_type.upper()}]\n")
        out.write(f"{'='*60}\n\n")
        source = "".join(cell.get("source", []))
        out.write(source)
        out.write("\n\n")
