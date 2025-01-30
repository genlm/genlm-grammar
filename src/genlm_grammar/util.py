import html
from IPython.display import HTML, display


def format_table(rows, headings=None):
    def fmt(x):
        if hasattr(x, "_repr_html_"):
            return x._repr_html_()
        elif hasattr(x, "_repr_svg_"):
            return x._repr_svg_()
        elif hasattr(x, "_repr_image_svg_xml"):
            return x._repr_image_svg_xml()
        else:
            return f"<pre>{html.escape(str(x))}</pre>"

    return (
        "<table>"
        + (
            '<tr style="font-weight: bold;">'
            + "".join(f"<td>{x}</td>" for x in headings)
            + "</tr>"
            if headings
            else ""
        )
        + "".join(
            "<tr>" + "".join(f"<td>{fmt(x)}</td>" for x in row) + " </tr>"
            for row in rows
        )
        + "</table>"
    )


def display_table(*args, **kwargs):
    return display(HTML(format_table(*args, **kwargs)))
