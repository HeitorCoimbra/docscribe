def generate_pdf_report(summaries: list[dict]) -> bytes:
    """Generate a PDF report with 2 leitos per page (A4).

    Each item in summaries must be a LeitoSummary-compatible dict:
    {"leito": str, "nome_paciente": str, "quadro_clinico": [...], "pendencias": [...], "condutas": [...]}
    """
    import re
    from io import BytesIO
    from datetime import datetime

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="DocScribe — Sumários de Passagem de Plantão",
    )

    styles = getSampleStyleSheet()

    doc_header_style = ParagraphStyle(
        "DocHeader",
        parent=styles["Normal"],
        fontSize=12,
        fontName="Helvetica-Bold",
        spaceAfter=2,
        textColor=colors.HexColor("#1a1a2e"),
    )
    date_style = ParagraphStyle(
        "DateStyle",
        parent=styles["Normal"],
        fontSize=9,
        fontName="Helvetica",
        spaceAfter=10,
        textColor=colors.HexColor("#666666"),
    )
    leito_title_style = ParagraphStyle(
        "LeitoTitle",
        parent=styles["Normal"],
        fontSize=12,
        fontName="Helvetica-Bold",
        spaceBefore=4,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Normal"],
        fontSize=10,
        fontName="Helvetica-Bold",
        spaceBefore=6,
        spaceAfter=2,
        textColor=colors.HexColor("#2d6a4f"),
    )
    item_style = ParagraphStyle(
        "Item",
        parent=styles["Normal"],
        fontSize=9,
        fontName="Helvetica",
        leading=13,
        leftIndent=12,
        spaceAfter=1,
    )

    # Sort summaries by leito number
    def _leito_key(d: dict):
        m = re.match(r"(\d+)", d.get("leito", ""))
        return int(m.group(1)) if m else 9999

    sorted_summaries = sorted(summaries, key=_leito_key)

    story = []

    # Document header
    date_str = datetime.now().strftime("%d/%m/%Y às %H:%M")
    story.append(
        Paragraph("DocScribe — Sumários de Passagem de Plantão", doc_header_style)
    )
    story.append(Paragraph(f"Gerado em {date_str}", date_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#2d6a4f"),
            spaceAfter=10,
        )
    )

    for i, leito_data in enumerate(sorted_summaries):
        leito_num = leito_data.get("leito", "?")
        nome = leito_data.get("nome_paciente", "—")

        story.append(
            Paragraph(f"Leito {leito_num} — {nome}", leito_title_style)
        )

        story.append(Paragraph("Quadro Clínico", section_style))
        for j, item in enumerate(leito_data.get("quadro_clinico") or ["—"], 1):
            story.append(Paragraph(f"{j}. {item}", item_style))

        story.append(Paragraph("Pendências", section_style))
        for j, item in enumerate(leito_data.get("pendencias") or ["—"], 1):
            story.append(Paragraph(f"{j}. {item}", item_style))

        story.append(Paragraph("Condutas", section_style))
        for item in leito_data.get("condutas") or ["—"]:
            story.append(Paragraph(f"• {item}", item_style))

        is_last = (i + 1) == len(sorted_summaries)
        is_second_on_page = (i + 1) % 2 == 0

        if not is_last:
            if is_second_on_page:
                story.append(PageBreak())
            else:
                story.append(Spacer(1, 10))
                story.append(
                    HRFlowable(
                        width="100%",
                        thickness=0.5,
                        color=colors.HexColor("#aaaaaa"),
                        spaceAfter=10,
                    )
                )

    doc.build(story)
    return buffer.getvalue()
