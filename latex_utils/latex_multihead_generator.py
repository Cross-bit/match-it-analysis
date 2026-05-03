from latex_utils.latex_table_generator import *

# ===================================
# DESCRIPTION
# ===================================
# Generator of latex tables from pandas dataframe.
#
#


class MultiHeaderLaTeXTableGenerator(LaTeXTableGenerator):
    def __init__(self, df, group_headers, group_size, column_width=1.5, column_specs=None):
        self.group_headers = group_headers
        self.group_size = group_size
        self.column_width = column_width
        self.column_specs = column_specs or [(2, 2)] * (len(df.columns) - 1)
        super().__init__(df, self.column_specs, self.column_width)

    def _default_column_format(self):
        specs = [
            "@{\\hspace{"+ str(self.column_width) +"cm}}D{.}{,}{" + f"{a}.{b}" + "}"
            for (a, b) in self.column_specs
        ]
        return "l" + "".join(specs)

    def generate_table(self, caption=None, label=None, note=None):
        lines = []
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append(f"\\begin{{tabular}}{{{self._column_format()}}}")
        lines.append("\\toprule")

        # First header: group titles
        group_row = [""] + [
            f"\\multicolumn{{{self.group_size}}}{{c}}{{{name}}}"
            for name in self.group_headers
        ]
        lines.append(" & ".join(group_row) + " \\\\")

        # Second header: sub-columns
        header_labels = [str(self.df.columns[0])] + list(self.df.columns[1:])
        header_cells = [f"\\textbf{{{header_labels[0]}}}"] + [
            f"\\mc{{\\textbf{{{col}}}}}" for col in header_labels[1:]
        ]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        for _, row in self.df.iterrows():
            row_data = [str(row[0])]
            for val in row[1:]:
                if isinstance(val, str):
                    row_data.append(f"\\mc{{{val}}}")
                else:
                    row_data.append(str(val))
            lines.append(" & ".join(row_data) + " \\\\")

        lines.append("\\bottomrule")

        if note:
            lines.append(
                f"\\multicolumn{{{1 + len(self.column_specs)}}}{{l}}{{\\footnotesize {note}}}"
            )

        lines.append("\\end{tabular}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)