#!/bin/python3
import pandas as pd

# ===================================
# DESCRIPTION
# ===================================
# Generator of latex tables from pandas dataframe.
#
#

class LaTeXTableGenerator:
    def __init__(self, df: pd.DataFrame, column_specs=None, column_width=1.5):
        self.df = df
        self.column_width = column_width
        # Define column specs like [(3,2), (1,2), (2,3)] → D{.}{,}{3.2} etc.
        self.column_specs = column_specs or [(2, 2)] * (len(df.columns) - 1)

    def _column_format(self):
        specs = [
            "@{\\hspace{" + str(self.column_width) + "cm}}D{.}{,}{" + f"{a}.{b}" + "}"
            for (a, b) in self.column_specs
        ]
        return "l" + "".join(specs)

    def generate_table(self, caption=None, label=None, note=None):
        lines = []
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append(f"\\begin{{tabular}}{{{self._column_format()}}}")
        lines.append("\\toprule")

        # First header row (merged with \mc)
        lines.append(" & " + " & ".join(["\\mc{}" for _ in self.column_specs]) + " \\\\")

        # Second header row
        header_labels = [str(self.df.columns[0])] + list(self.df.columns[1:])
        header_cells = [
            f"\\pulrad{{\\textbf{{{header_labels[0]}}}}}"
        ] + [
            f"\\mc{{\\pulrad{{\\textbf{{{label}}}}}}}" for label in header_labels[1:]
        ]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        # Table content
        for _, row in self.df.iterrows():
            row_data = [str(row[0])]
            for val in row[1:]:
                if isinstance(val, str):
                    row_data.append(f"\\mc{{{val}}}")
                else:
                    row_data.append(str(val))
            lines.append(" & ".join(row_data) + " \\\\")

        lines.append("\\bottomrule")

        # Optional note (e.g., footnote)
        if note:
            num_cols = 1 + len(self.column_specs)
            lines.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\footnotesize {note}}}")

        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)


class LaTeXTableGeneratorSIUnitx:
    def __init__(self, df: pd.DataFrame, column_specs=None, column_width=1.5):
        self.df = df
        self.column_width = column_width
        self.column_specs = column_specs or [(2, 2)] * (len(df.columns) - 1)

    def _column_format(self):
        specs = [
            f"S[table-format={a}.{b}]"  # , table-column-width={self.column_width}cm]
            for (a, b) in self.column_specs
        ]
        return "l " + " ".join(specs)

    def generate_table(self, caption=None, label=None, note=None, cell_bold_fn=None, include_index_header: bool = True):
        lines = []

        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append(f"\\begin{{tabular}}{{{self._column_format()}}}")
        lines.append("\\toprule")

        # Header
        if include_index_header:
            first_col_name = str(self.df.columns[0])  # první sloupec podle POZICE
            rest = " & ".join([f"\\textbf{{{col}}}" for col in self.df.columns[1:]])
            lines.append(f"\\textbf{{{first_col_name}}} & {rest} \\\\")
        else:
            lines.append(" & " + " & ".join([f"\\textbf{{{col}}}" for col in self.df.columns[1:]]) + " \\\\")
        lines.append("\\midrule")

        for row_idx, row in self.df.iterrows():
            row_label = str(row.iloc[0])  # <<< FIX: první buňka beru pozicově

            def format_cell(col_idx, val):
                if isinstance(val, str):
                    return val
                bold = cell_bold_fn(row_idx, col_idx, val) if cell_bold_fn else False
                return f"\\bfseries\\num{{{val}}}" if bold else f"\\num{{{val}}}"

            row_data = [row_label] + [
                format_cell(col_idx, val) for col_idx, val in enumerate(row.iloc[1:], start=1)  # <<< FIX: i zbytek pozicově
            ]
            lines.append(" & ".join(row_data) + " \\\\")

        lines.append("\\bottomrule")

        if note:
            num_cols = 1 + len(self.column_specs)
            lines.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\footnotesize {note}}}")

        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)

#
# Example usage
#
#data = {
#    "Efekt": ["Abs. člen", "Pohlaví (muž)", "Výška (cm)"],
#    "Odhad": [-10.01, 9.89, 0.78],
#    "Směrod. chyba": [1.01, 5.98, 0.12],
#    "P-hodnota": ["\\mc{---}", 0.098, "<0,001"]
#}
#
#df = pd.DataFrame(data)
#
#generator = LaTeXTableGenerator(
#    df,
#    column_specs=[(3, 2), (1, 2), (2, 3)]  # odpovídá D{.}{,}{3.2} atd.
#)
#
#latex_code = generator.generate_table(
#    caption="Maximálně věrohodné odhady v~modelu M.",
#    label="tab03:Nejaka",
#    note="\\textit{Pozn:} $^a$ Směrodatná chyba odhadu metodou Monte Carlo."
#)
#
## Print or save the result
#print(latex_code)