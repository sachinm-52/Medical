"""
PDF Medical Report Generator for MedVision AI.
Uses ReportLab to generate a radiology AI analysis summary report.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from backend.app.config import settings

REPORTS_DIR = settings.STORAGE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class PDFService:
    def generate_report(
        self,
        analysis_id: int,
        model_name: str,
        architecture: str,
        dataset_name: str,
        overall_assessment: str,
        predictions: list,
        image_path: str,
        gradcam_path: Optional[str] = None,
        created_at: Optional[datetime.datetime] = None
    ) -> str:
        """
        Generates a PDF analysis report and returns the absolute file path.
        """
        timestamp_str = (created_at or datetime.datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S UTC")
        pdf_filename = f"medvision_report_{analysis_id}.pdf"
        pdf_path = REPORTS_DIR / pdf_filename

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0369a1"),
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#64748b")
        )
        section_style = ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=12,
            spaceAfter=6
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#991b1b")
        )

        elements = []

        # Header
        elements.append(Paragraph("<b>MedVision AI</b> — Explainable Chest X-ray Report", title_style))
        elements.append(Paragraph(f"Analysis ID: #{analysis_id} &nbsp;|&nbsp; Generated: {timestamp_str}", subtitle_style))
        elements.append(Spacer(1, 10))

        # Medical Disclaimer Callout Box
        disclaimer_text = (
            "<b>IMPORTANT CLINICAL DISCLAIMER:</b> MedVision AI is an educational and research deep learning tool. "
            "Model predictions and Grad-CAM visualizations represent algorithmic patterns and are NOT intended as "
            "definitive medical diagnoses or treatment advice. Always consult a qualified board-certified radiologist."
        )
        disclaimer_table = Table(
            [[Paragraph(disclaimer_text, disclaimer_style)]],
            colWidths=[540]
        )
        disclaimer_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#f87171")),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(disclaimer_table)
        elements.append(Spacer(1, 12))

        # Metadata Section
        elements.append(Paragraph("<b>Case & Model Specifications</b>", section_style))
        meta_data = [
            ["Model Pipeline:", model_name, "Architecture:", architecture],
            ["Training Dataset:", dataset_name, "Inference Device:", "PyTorch Engine"],
            ["Overall Assessment:", overall_assessment, "Verification Status:", "Automated Screening"]
        ]
        meta_table = Table(meta_data, colWidths=[120, 150, 120, 150])
        meta_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#334155")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 10))

        # Disease Findings Table
        elements.append(Paragraph("<b>Disease Detection Findings</b>", section_style))
        table_rows = [["Disease / Pathology", "Score / Probability", "Threshold", "Finding Status", "Confidence"]]
        for p in predictions[:14]:  # Top or all 14 findings
            score = p.get("score", 0.0)
            threshold = p.get("threshold", 0.5)
            is_pos = p.get("is_positive", False)
            status_text = "ELEVATED FINDING" if is_pos else "Within Normal Limit"
            confidence = p.get("confidence_level", "Moderate")

            table_rows.append([
                p.get("disease", "Unknown"),
                f"{score:.1%}",
                f"{threshold:.1%}",
                status_text,
                confidence
            ])

        findings_table = Table(table_rows, colWidths=[150, 95, 80, 125, 90])
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]
        # Highlight positive rows
        for row_idx, p in enumerate(predictions[:14], start=1):
            if p.get("is_positive", False):
                table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#fff1f2")))
                table_style.append(("TEXTCOLOR", (3, row_idx), (3, row_idx), colors.HexColor("#e11d48")))
                table_style.append(("FONTNAME", (3, row_idx), (3, row_idx), "Helvetica-Bold"))

        findings_table.setStyle(TableStyle(table_style))
        elements.append(findings_table)
        elements.append(Spacer(1, 14))

        # Explainability & Visual Evidence Section
        elements.append(Paragraph("<b>Radiographic Visuals & Explainable AI (Grad-CAM)</b>", section_style))
        img_elements = []
        if Path(image_path).exists():
            img_elements.append(RLImage(image_path, width=2.4*inch, height=2.4*inch))
        if gradcam_path and Path(gradcam_path).exists():
            img_elements.append(RLImage(gradcam_path, width=2.4*inch, height=2.4*inch))

        if img_elements:
            img_table = Table([img_elements], colWidths=[270] * len(img_elements))
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            elements.append(KeepTogether([img_table]))

        doc.build(elements)
        return str(pdf_path)

pdf_service = PDFService()
