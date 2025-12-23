# Dependencies
from typing import Any
from typing import List
from pathlib import Path
from typing import Optional
from datetime import datetime
from reportlab.lib import colors
from utils.logger import get_logger
from config.settings import settings
from reportlab.platypus import Table
from reportlab.lib.units import inch
from reportlab.platypus import Spacer
from reportlab.platypus import Paragraph
from reportlab.platypus import PageBreak
from reportlab.platypus import TableStyle
from config.schemas import AnalysisResult
from config.constants import FinalDecision
from reportlab.lib.pagesizes import LETTER
from utils.helpers import generate_unique_id
from config.constants import EvidenceStrength
from config.schemas import BatchAnalysisResult
from reportlab.lib.styles import ParagraphStyle 
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from features.detailed_result_maker import DetailedResultMaker


# Setup Logging
logger = get_logger(__name__)


class PDFReporter:
    """
    PDF Report Generator

    Guarantees:
    -----------
    - FinalDecision is authoritative
    - Evidence-first explanations
    - Metrics are informational only
    - Audit-safe and regulator-ready
    """
    COLOR_PRIMARY = colors.HexColor('#0D47A1')
    COLOR_SUCCESS = colors.HexColor('#1B5E20')
    COLOR_WARNING = colors.HexColor('#E65100')
    COLOR_DANGER  = colors.HexColor('#B71C1C')
    COLOR_HEADER  = colors.HexColor('#1565C0')
    COLOR_ALT_ROW = colors.HexColor('#F5F5F5')

    def __init__(self):
        self.detailed_maker = DetailedResultMaker()
        self.styles         = self._build_styles()
        logger.debug("PDFReporter initialized")


    def export_single(self, result: AnalysisResult, output_dir: Optional[Path] = None) -> Path:
        """
        Export single image PDF report
        """
        output_dir  = output_dir or settings.REPORTS_DIR
        output_dir.mkdir(parents = True, exist_ok = True)

        filename    = f"ai_screener_report_{generate_unique_id()}.pdf"
        output_path = output_dir / filename

        logger.info(f"Generating single image PDF: {filename}")

        doc         = SimpleDocTemplate(str(output_path),
                                        pagesize     = LETTER,
                                        rightMargin  = 30,
                                        leftMargin   = 30,
                                        topMargin    = 20,
                                        bottomMargin = 35,
                                       )

        story       = list()

        self._add_header(story, "AI Image Analysis Report")

        self._add_single_executive_summary(story, result)
        
        story.append(PageBreak())
        
        self._add_evidence_section(story, result)
        
        story.append(PageBreak())
        
        self._add_metrics_section(story, result)
        self._add_footer(story)

        doc.build(story)
        
        return output_path


    def export_batch(self, batch_result: BatchAnalysisResult, output_dir: Optional[Path] = None) -> Path:
        """
        Export batch PDF report
        """
        output_dir  = output_dir or settings.REPORTS_DIR
        output_dir.mkdir(parents = True, exist_ok = True)

        filename    = f"ai_screener_batch_{generate_unique_id()}.pdf"
        output_path = output_dir / filename

        logger.info(f"Generating batch PDF: {filename}")

        doc         = SimpleDocTemplate(str(output_path),
                                        pagesize     = LETTER,
                                        rightMargin  = 30,
                                        leftMargin   = 30,
                                        topMargin    = 20,
                                        bottomMargin = 35,
                                       )

        story       = list()

        self._add_header(story, "Batch Image Analysis Report")
        self._add_batch_summary(story, batch_result)
        story.append(PageBreak())

        for idx, result in enumerate(batch_result.results, 1):
            self._add_single_executive_summary(story, result, index=idx)
            self._add_evidence_section(story, result)
            self._add_metrics_section(story, result)

            if (idx < len(batch_result.results)):
                story.append(PageBreak())

        self._add_footer(story)
        doc.build(story)

        return output_path


    def _add_header(self, story, title: str):
        story.append(Paragraph("AI IMAGE SCREENER", self.styles['Title']))
        story.append(Paragraph(title, self.styles['Subtitle']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Version: {settings.VERSION}", self.styles['Meta']))
        story.append(Spacer(1, 12))


    def _add_single_executive_summary(self, story, result: AnalysisResult, index: Optional[int] = None):
        title = "Executive Summary"

        if index:
            title += f" — Image {index}"

        story.append(Paragraph(title, self.styles['Section']))

        decision = result.final_decision.value if result.final_decision else "UNDECIDED"
        color    = self._decision_color(result.final_decision)

        table    = Table([["Final Decision", decision],
                          ["Confidence", f"{result.confidence}%"],
                          ["Explanation", result.decision_explanation or "—"],
                         ],
                         colWidths = [140, 390]
                        )

        table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), self.COLOR_HEADER),
                                   ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                   ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                   ('BACKGROUND', (0, 1), (-1, -1), color),
                                   ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                   ('LEFTPADDING', (0, 0), (-1, -1), 8),
                                   ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                                 ])
                      )

        story.append(table)
        story.append(Spacer(1, 10))


    def _add_evidence_section(self, story, result: AnalysisResult):
        story.append(Paragraph("Evidence Assessment", self.styles['Section']))

        if not result.evidence:
            story.append(Paragraph("No declarative evidence detected. Decision derived from Tier-1 metrics.", self.styles['Body']))
            return

        rows = [["Source", "Direction", "Strength", "Confidence", "Finding"]]

        for e in result.evidence:
            rows.append([e.source.value,
                         e.direction.value,
                         e.strength.value,
                         f"{e.confidence:.2f}" if e.confidence else "N/A",
                         e.finding
                       ])

        table = Table(rows, colWidths = [70, 80, 80, 70, 230])
        table.setStyle(self._standard_table_style())

        story.append(table)
        story.append(Spacer(1, 10))


    def _add_metrics_section(self, story, result: AnalysisResult):
        story.append(Paragraph("Metric Signals (Informational)", self.styles['Section']))

        rows = [["Metric", "Score", "Confidence", "Notes"]]

        for mt, mr in result.metric_results.items():
            rows.append([
                self.detailed_maker.metric_display_names.get(mt, mt.value),
                f"{mr.score:.3f}",
                f"{mr.confidence:.3f}" if mr.confidence else "N/A",
                ", ".join(self.detailed_maker.extract_key_findings(mt, mr))
            ])

        table = Table(rows, colWidths=[180, 70, 80, 210])
        table.setStyle(self._standard_table_style())

        story.append(table)
        story.append(Spacer(1, 10))


    def _add_batch_summary(self, story, batch_result: BatchAnalysisResult):
        story.append(Paragraph("Batch Decision Summary", self.styles['Section']))

        rows = [
            ["Total Images", batch_result.total_images],
            ["Processed", batch_result.processed],
            ["Failed", batch_result.failed],
            ["Success Rate", f"{batch_result.summary.get('success_rate', 0)}%"],
        ]

        for decision in FinalDecision:
            rows.append([
                decision.value,
                batch_result.summary.get(decision.value, 0)
            ])

        table = Table(rows, colWidths=[220, 310])
        table.setStyle(self._standard_table_style())

        story.append(table)
        story.append(Spacer(1, 10))


    def _add_footer(self, story):
        story.append(Spacer(1, 15))
        story.append(Paragraph(
            "DISCLAIMER: Metric scores are non-decisional. "
            "Final decisions are evidence- and policy-based.",
            self.styles['Footer']
        ))


    # ------------------------------------------------------------------
    # STYLES & HELPERS
    # ------------------------------------------------------------------

    def _build_styles(self):
        styles = getSampleStyleSheet()

        styles.add(ParagraphStyle(
            name='Title',
            fontSize=18,
            alignment=1,
            textColor=self.COLOR_PRIMARY,
            fontName='Helvetica-Bold'
        ))

        styles.add(ParagraphStyle(
            name='Subtitle',
            fontSize=12,
            alignment=1,
            spaceAfter=6
        ))

        styles.add(ParagraphStyle(
            name='Meta',
            fontSize=8,
            alignment=1,
            spaceAfter=10,
            textColor=colors.grey
        ))

        styles.add(ParagraphStyle(
            name='Section',
            fontSize=13,
            fontName='Helvetica-Bold',
            spaceBefore=10,
            spaceAfter=6
        ))

        styles.add(ParagraphStyle(
            name='Body',
            fontSize=9,
            spaceAfter=6
        ))

        styles.add(ParagraphStyle(
            name='Footer',
            fontSize=7,
            alignment=1,
            textColor=colors.grey
        ))

        return styles


    def _standard_table_style(self):
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_HEADER),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLOR_ALT_ROW]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ])


    def _decision_color(self, decision: Optional[FinalDecision]):
        if decision == FinalDecision.CONFIRMED_AI_GENERATED:
            return colors.HexColor('#FFEBEE')
        if decision == FinalDecision.SUSPICIOUS_AI_LIKELY:
            return colors.HexColor('#FFF3E0')
        if decision == FinalDecision.AUTHENTIC_BUT_REVIEW:
            return colors.HexColor('#E3F2FD')
        if decision == FinalDecision.MOSTLY_AUTHENTIC:
            return colors.HexColor('#E8F5E9')
        return colors.white