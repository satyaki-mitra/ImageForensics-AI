# Dependencies
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from utils.logger import get_logger
from config.settings import settings
from reportlab.platypus import Table, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import TableStyle
from config.schemas import AnalysisResult
from utils.helpers import generate_unique_id
from config.constants import DetectionStatus
from config.schemas import BatchAnalysisResult
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate
from features.detailed_result_maker import DetailedResultMaker
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import textwrap


# Setup Logging
logger = get_logger(__name__)


class PDFReporter:
    """
    Professional-Grade PDF Report Generator for AI Image Analysis
    
    Features:
    ---------
    - Comprehensive single image reports with full forensic details
    - Multi-page batch reports with executive summary
    - Enhanced visual hierarchy and color coding
    - Detailed metric breakdowns with explanations
    - Professional formatting and layout
    - Statistical summaries and insights
    """
    
    # Enhanced Color Scheme
    COLOR_PRIMARY = colors.HexColor('#0D47A1')        # Deep Blue
    COLOR_SUCCESS = colors.HexColor('#1B5E20')        # Dark Green
    COLOR_WARNING = colors.HexColor('#E65100')        # Deep Orange
    COLOR_DANGER = colors.HexColor('#B71C1C')         # Dark Red
    COLOR_INFO = colors.HexColor('#01579B')           # Light Blue
    COLOR_NEUTRAL = colors.HexColor('#424242')        # Dark Grey
    COLOR_HEADER_BG = colors.HexColor('#1565C0')      # Blue
    COLOR_SUBHEADER_BG = colors.HexColor('#1976D2')   # Lighter Blue
    COLOR_ALT_ROW = colors.HexColor('#F5F5F5')        # Light Grey
    COLOR_LIGHT_BLUE = colors.HexColor('#E3F2FD')     # Very Light Blue
    COLOR_LIGHT_GREEN = colors.HexColor('#E8F5E9')    # Very Light Green
    COLOR_LIGHT_ORANGE = colors.HexColor('#FFF3E0')   # Very Light Orange
    COLOR_LIGHT_RED = colors.HexColor('#FFEBEE')      # Very Light Red
    
    def __init__(self):
        self.detailed_maker = DetailedResultMaker()
        self.styles = self._build_styles()
        logger.debug("Enhanced PDFReporter initialized")

    def export_single(self, result: AnalysisResult, output_dir: Optional[Path] = None) -> Path:
        """Export comprehensive single image analysis report"""
        output_dir = output_dir or settings.REPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_id = generate_unique_id()
        filename = f"ai_screener_report_{report_id}.pdf"
        output_path = output_dir / filename
        
        logger.info(f"Generating comprehensive single image PDF: {filename}")
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=LETTER,
            rightMargin=30,
            leftMargin=30,
            topMargin=20,
            bottomMargin=35
        )
        
        story = []
        self._add_professional_header(story, "AI Image Analysis Report")
        self._add_executive_summary_single(story, result)
        story.append(PageBreak())
        self._add_detailed_metrics_analysis(story, result)
        story.append(PageBreak())
        self._add_forensic_breakdown(story, result)
        self._add_recommendations(story, result)
        self._add_professional_footer(story)
        
        doc.build(story, onFirstPage=self._add_watermark, onLaterPages=self._add_watermark)
        logger.info(f"Single image report generated: {output_path}")
        return output_path

    def export_batch(self, batch_result: BatchAnalysisResult, output_dir: Optional[Path] = None) -> Path:
        """Export comprehensive batch analysis report"""
        output_dir = output_dir or settings.REPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_id = generate_unique_id()
        filename = f"ai_screener_report_{report_id}.pdf"
        output_path = output_dir / filename
        
        num_images = len(batch_result.results)
        logger.info(f"Generating batch PDF report: {filename} ({num_images} images)")
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=LETTER,
            rightMargin=30,
            leftMargin=30,
            topMargin=20,
            bottomMargin=35
        )
        
        story = []
        self._add_professional_header(story, "Batch Image Analysis Report")
        self._add_batch_executive_summary(story, batch_result)
        story.append(PageBreak())
        self._add_batch_overview_table(story, batch_result.results)
        story.append(PageBreak())
        self._add_batch_metrics_analysis(story, batch_result.results)
        story.append(PageBreak())
        self._add_individual_results_summary(story, batch_result.results)
        self._add_batch_recommendations(story, batch_result)
        self._add_professional_footer(story)
        
        doc.build(story, onFirstPage=self._add_watermark, onLaterPages=self._add_watermark)
        logger.info(f"Batch report generated: {output_path}")
        return output_path

    def _build_styles(self):
        """Build comprehensive style definitions"""
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='ReportTitle',
            fontSize=18,
            textColor=self.COLOR_PRIMARY,
            alignment=TA_CENTER,
            spaceAfter=4,
            spaceBefore=2,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='ReportSubtitle',
            fontSize=10,
            textColor=self.COLOR_NEUTRAL,
            alignment=TA_CENTER,
            spaceAfter=6,
            fontName='Helvetica'
        ))
        
        styles.add(ParagraphStyle(
            name='SectionTitle',
            fontSize=13,
            textColor=self.COLOR_PRIMARY,
            spaceBefore=10,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            fontSize=11,
            textColor=self.COLOR_PRIMARY,
            spaceBefore=8,
            spaceAfter=5,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='SubHeader',
            fontSize=9.5,
            textColor=self.COLOR_PRIMARY,
            spaceBefore=5,
            spaceAfter=3,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBodyText',
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
            spaceAfter=6
        ))
        
        styles.add(ParagraphStyle(
            name='TableCell',
            fontSize=8,
            leading=10
        ))
        
        styles.add(ParagraphStyle(
            name='TableCellSmall',
            fontSize=7.5,
            leading=9
        ))
        
        styles.add(ParagraphStyle(
            name='TableHeader',
            fontSize=8.5,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            leading=10,
            alignment=TA_CENTER
        ))
        
        styles.add(ParagraphStyle(
            name='Footer',
            fontSize=7.5,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceAfter=2
        ))
        
        styles.add(ParagraphStyle(
            name='Timestamp',
            fontSize=8,
            textColor=self.COLOR_NEUTRAL,
            alignment=TA_CENTER,
            spaceAfter=8
        ))
        
        return styles

    def _add_watermark(self, canvas, doc):
        """Add professional watermark"""
        canvas.saveState()
        canvas.setFont('Helvetica-Bold', 70)
        canvas.setFillColorRGB(0.85, 0.85, 0.85, alpha=0.15)
        canvas.rotate(45)
        canvas.drawString(2.5*inch, -0.5*inch, "AI IMAGE SCREENER")
        canvas.restoreState()

    def _add_professional_header(self, story, title: str):
        """Professional header with branding"""
        story.append(Paragraph("🔍 AI IMAGE SCREENER", self.styles['ReportTitle']))
        story.append(Spacer(1, 3))
        
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Version: {settings.VERSION}"
        story.append(Paragraph(timestamp_text, self.styles['Timestamp']))
        
        story.append(Paragraph(title, self.styles['SectionTitle']))
        story.append(Spacer(1, 10))

    def _add_executive_summary_single(self, story, result: AnalysisResult):
        """Executive summary for single image"""
        story.append(Paragraph("Executive Summary", self.styles['SectionTitle']))
        story.append(Spacer(1, 5))
        
        # Key findings box
        status_color = self.COLOR_DANGER if result.status == DetectionStatus.REVIEW_REQUIRED else self.COLOR_SUCCESS
        status_bg = self.COLOR_LIGHT_RED if result.status == DetectionStatus.REVIEW_REQUIRED else self.COLOR_LIGHT_GREEN
        status_text = "⚠️ REVIEW REQUIRED" if result.status == DetectionStatus.REVIEW_REQUIRED else "✅ LIKELY AUTHENTIC"
        
        key_findings = [
            [Paragraph("<b>Overall Assessment</b>", self.styles['TableHeader'])],
            [Paragraph(f"<font size=12 color='{status_color.hexval()}'><b>{status_text}</b></font>", self.styles['CustomBodyText'])],
            [Paragraph(f"<b>Confidence:</b> {result.confidence}%", self.styles['CustomBodyText'])],
            [Paragraph(f"<b>Overall Score:</b> {result.overall_score:.4f}", self.styles['CustomBodyText'])]
        ]
        
        findings_table = Table(key_findings, colWidths=[530])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_INFO),
            ('BACKGROUND', (0, 1), (-1, -1), status_bg),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 1.5, self.COLOR_PRIMARY)
        ]))
        story.append(findings_table)
        story.append(Spacer(1, 12))
        
        # Image information
        story.append(Paragraph("Image Information", self.styles['SectionHeader']))
        
        info_data = [
            [Paragraph("<b>Property</b>", self.styles['TableHeader']), 
             Paragraph("<b>Value</b>", self.styles['TableHeader'])],
            [Paragraph("Filename", self.styles['TableCell']), 
             Paragraph(result.filename, self.styles['TableCell'])],
            [Paragraph("Dimensions", self.styles['TableCell']), 
             Paragraph(f"{result.image_size[0]} × {result.image_size[1]} pixels", self.styles['TableCell'])],
            [Paragraph("Aspect Ratio", self.styles['TableCell']), 
             Paragraph(f"{result.image_size[0]/result.image_size[1]:.2f}:1", self.styles['TableCell'])],
            [Paragraph("Processing Time", self.styles['TableCell']), 
             Paragraph(f"{result.processing_time:.3f} seconds", self.styles['TableCell'])],
            [Paragraph("Analysis Date", self.styles['TableCell']), 
             Paragraph(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.styles['TableCell'])]
        ]
        
        info_table = Table(info_data, colWidths=[180, 350])
        info_table.setStyle(self._get_standard_table_style(len(info_data)))
        story.append(info_table)
        story.append(Spacer(1, 12))
        
        # Detection signals summary
        story.append(Paragraph("Detection Signals Summary", self.styles['SectionHeader']))
        
        flagged = sum(1 for s in result.signals if s.status.value == 'flagged')
        warning = sum(1 for s in result.signals if s.status.value == 'warning')
        passed = sum(1 for s in result.signals if s.status.value == 'passed')
        
        signals_data = [
            [Paragraph("<b>Status</b>", self.styles['TableHeader']), 
             Paragraph("<b>Count</b>", self.styles['TableHeader']),
             Paragraph("<b>Percentage</b>", self.styles['TableHeader'])],
            [Paragraph("🔴 Flagged", self.styles['TableCell']), 
             Paragraph(f"<font color='red'><b>{flagged}</b></font>", self.styles['TableCell']),
             Paragraph(f"{flagged/len(result.signals)*100:.1f}%", self.styles['TableCell'])],
            [Paragraph("🟡 Warning", self.styles['TableCell']), 
             Paragraph(f"<font color='orange'><b>{warning}</b></font>", self.styles['TableCell']),
             Paragraph(f"{warning/len(result.signals)*100:.1f}%", self.styles['TableCell'])],
            [Paragraph("🟢 Passed", self.styles['TableCell']), 
             Paragraph(f"<font color='green'><b>{passed}</b></font>", self.styles['TableCell']),
             Paragraph(f"{passed/len(result.signals)*100:.1f}%", self.styles['TableCell'])]
        ]
        
        signals_table = Table(signals_data, colWidths=[200, 165, 165])
        signals_table.setStyle(self._get_standard_table_style(len(signals_data)))
        story.append(signals_table)

    def _add_detailed_metrics_analysis(self, story, result: AnalysisResult):
        """Comprehensive metrics analysis"""
        story.append(Paragraph("Detailed Metrics Analysis", self.styles['SectionTitle']))
        story.append(Spacer(1, 8))
        
        # All detection signals with full details
        story.append(Paragraph("Detection Signals Breakdown", self.styles['SectionHeader']))
        
        signal_data = [
            [Paragraph("<b>Metric</b>", self.styles['TableHeader']),
             Paragraph("<b>Score</b>", self.styles['TableHeader']),
             Paragraph("<b>Status</b>", self.styles['TableHeader']),
             Paragraph("<b>Explanation</b>", self.styles['TableHeader'])]
        ]
        
        for signal in result.signals:
            status_badge = self._get_status_badge_html(signal.status.value)
            
            # Wrap long explanations
            explanation = signal.explanation
            if len(explanation) > 120:
                explanation = explanation[:120] + "..."
            
            signal_data.append([
                Paragraph(f"<b>{signal.name}</b>", self.styles['TableCell']),
                Paragraph(f"{signal.score:.4f}", self.styles['TableCell']),
                Paragraph(status_badge, self.styles['TableCell']),
                Paragraph(explanation, self.styles['TableCellSmall'])
            ])
        
        signal_table = Table(signal_data, colWidths=[120, 60, 80, 270])
        signal_table.setStyle(self._get_signal_table_style(len(signal_data)))
        story.append(signal_table)

    def _add_forensic_breakdown(self, story, result: AnalysisResult):
        """Detailed forensic analysis breakdown"""
        story.append(Paragraph("Forensic Analysis Breakdown", self.styles['SectionTitle']))
        story.append(Spacer(1, 8))
        
        for metric_type, metric_result in result.metric_results.items():
            metric_name = self.detailed_maker.metric_display_names.get(metric_type, metric_type.value)
            details = metric_result.details or {}
            
            # Skip if error
            if 'error' in details:
                continue
            
            story.append(Paragraph(metric_name, self.styles['SectionHeader']))
            
            # Metric overview
            overview_data = [
                [Paragraph("<b>Property</b>", self.styles['TableHeader']), 
                 Paragraph("<b>Value</b>", self.styles['TableHeader'])],
                [Paragraph("Score", self.styles['TableCell']), 
                 Paragraph(f"<b>{metric_result.score:.4f}</b>", self.styles['TableCell'])],
                [Paragraph("Confidence", self.styles['TableCell']), 
                 Paragraph(f"{metric_result.confidence:.4f}" if metric_result.confidence else "N/A", self.styles['TableCell'])],
                [Paragraph("Status", self.styles['TableCell']), 
                 Paragraph(self._get_metric_status_html(metric_result.score), self.styles['TableCell'])]
            ]
            
            overview_table = Table(overview_data, colWidths=[130, 400])
            overview_table.setStyle(self._get_standard_table_style(len(overview_data)))
            story.append(overview_table)
            story.append(Spacer(1, 5))
            
            # Detailed parameters
            if details and len(details) > 0:
                story.append(Paragraph("Detailed Parameters:", self.styles['SubHeader']))
                
                param_data = [[Paragraph("<b>Parameter</b>", self.styles['TableHeader']), 
                              Paragraph("<b>Value</b>", self.styles['TableHeader'])]]
                
                for key, value in details.items():
                    if key in ['error', 'reason']:
                        continue
                    
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key not in ['reason', 'error']:
                                formatted_value = self._format_value(sub_value)
                                param_data.append([
                                    Paragraph(f"  └─ {sub_key}", self.styles['TableCellSmall']),
                                    Paragraph(formatted_value, self.styles['TableCellSmall'])
                                ])
                    else:
                        formatted_value = self._format_value(value)
                        param_data.append([
                            Paragraph(key, self.styles['TableCell']),
                            Paragraph(formatted_value, self.styles['TableCell'])
                        ])
                
                param_table = Table(param_data, colWidths=[200, 330])
                param_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_SUBHEADER_BG),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLOR_ALT_ROW]),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
                ]))
                story.append(param_table)
            
            story.append(Spacer(1, 8))

    def _add_recommendations(self, story, result: AnalysisResult):
        """Add actionable recommendations"""
        story.append(Paragraph("Recommendations & Next Steps", self.styles['SectionTitle']))
        story.append(Spacer(1, 8))
        
        if result.status == DetectionStatus.REVIEW_REQUIRED:
            rec_text = """
            <b>⚠️ MANUAL REVIEW REQUIRED</b><br/>
            This image has been flagged for manual review based on multiple detection signals. 
            Recommended actions:<br/>
            • Conduct visual inspection by trained personnel<br/>
            • Cross-reference with source verification<br/>
            • Consider additional forensic analysis if high stakes<br/>
            • Document findings for audit trail
            """
            rec_color = self.COLOR_LIGHT_RED
            border_color = self.COLOR_DANGER
        else:
            rec_text = """
            <b>✅ NO IMMEDIATE ACTION REQUIRED</b><br/>
            This image appears to be authentic based on current analysis. However:<br/>
            • Continue monitoring for evolving AI techniques<br/>
            • Consider periodic re-screening for critical assets<br/>
            • Maintain chain of custody documentation<br/>
            • Stay updated on latest detection methodologies
            """
            rec_color = self.COLOR_LIGHT_GREEN
            border_color = self.COLOR_SUCCESS
        
        rec_table = Table([[Paragraph(rec_text, self.styles['CustomBodyText'])]], colWidths=[530])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), rec_color),
            ('BOX', (0, 0), (-1, -1), 2, border_color),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12)
        ]))
        story.append(rec_table)

    def _add_batch_executive_summary(self, story, batch_result: BatchAnalysisResult):
        """Executive summary for batch analysis"""
        story.append(Paragraph("Executive Summary", self.styles['SectionTitle']))
        story.append(Spacer(1, 8))
        
        # Key metrics
        summary_data = [
            [Paragraph("<b>Metric</b>", self.styles['TableHeader']), 
             Paragraph("<b>Value</b>", self.styles['TableHeader']),
             Paragraph("<b>Details</b>", self.styles['TableHeader'])],
            [Paragraph("Total Images", self.styles['TableCell']), 
             Paragraph(f"<b>{batch_result.total_images}</b>", self.styles['TableCell']),
             Paragraph("Images submitted for analysis", self.styles['TableCellSmall'])],
            [Paragraph("Successfully Processed", self.styles['TableCell']), 
             Paragraph(f"<font color='green'><b>{batch_result.processed}</b></font>", self.styles['TableCell']),
             Paragraph(f"{batch_result.summary.get('success_rate', 0)}% success rate", self.styles['TableCellSmall'])],
            [Paragraph("Failed", self.styles['TableCell']), 
             Paragraph(f"<font color='red'><b>{batch_result.failed}</b></font>", self.styles['TableCell']),
             Paragraph("Processing errors encountered", self.styles['TableCellSmall'])],
            [Paragraph("Likely Authentic", self.styles['TableCell']), 
             Paragraph(f"<font color='green'><b>{batch_result.summary.get('likely_authentic', 0)}</b></font>", self.styles['TableCell']),
             Paragraph("Images passing authenticity checks", self.styles['TableCellSmall'])],
            [Paragraph("Review Required", self.styles['TableCell']), 
             Paragraph(f"<font color='red'><b>{batch_result.summary.get('review_required', 0)}</b></font>", self.styles['TableCell']),
             Paragraph("Images flagged for manual review", self.styles['TableCellSmall'])],
            [Paragraph("Average Score", self.styles['TableCell']), 
             Paragraph(f"<b>{batch_result.summary.get('avg_score', 0):.4f}</b>", self.styles['TableCell']),
             Paragraph("Mean authenticity score across batch", self.styles['TableCellSmall'])],
            [Paragraph("Average Processing Time", self.styles['TableCell']), 
             Paragraph(f"<b>{batch_result.summary.get('avg_proc_time', 0):.3f}s</b>", self.styles['TableCell']),
             Paragraph("Per-image processing duration", self.styles['TableCellSmall'])],
        ]
        
        summary_table = Table(summary_data, colWidths=[150, 130, 250])
        summary_table.setStyle(self._get_standard_table_style(len(summary_data)))
        story.append(summary_table)

    def _add_batch_overview_table(self, story, results: List[AnalysisResult]):
        """Comprehensive batch overview"""
        story.append(Paragraph("Batch Overview Matrix", self.styles['SectionTitle']))
        story.append(Spacer(1, 8))
        
        header = [
            Paragraph("<b>#</b>", self.styles['TableHeader']),
            Paragraph("<b>Filename</b>", self.styles['TableHeader']),
            Paragraph("<b>Image Size</b>", self.styles['TableHeader']),
            Paragraph("<b>Score</b>", self.styles['TableHeader']),
            Paragraph("<b>Status</b>", self.styles['TableHeader']),
            Paragraph("<b>Top Signal</b>", self.styles['TableHeader']),
            Paragraph("<b>Time(s)</b>", self.styles['TableHeader'])
        ]
        
        data = [header]
        
        for idx, result in enumerate(results, 1):
            top_signal = max(result.signals, key=lambda s: s.score)
            status_badge = self._get_status_badge_short(result.status.value)
            
            data.append([
                Paragraph(str(idx), self.styles['TableCell']),
                Paragraph(result.filename, self.styles['TableCellSmall']),
                Paragraph(f"{result.image_size[0]}×{result.image_size[1]}", self.styles['TableCellSmall']),
                Paragraph(f"<b>{result.overall_score:.3f}</b>", self.styles['TableCell']),
                Paragraph(status_badge, self.styles['TableCellSmall']),
                Paragraph(f"{top_signal.name}: {top_signal.score:.2f}", self.styles['TableCellSmall']),
                Paragraph(f"{result.processing_time:.2f}", self.styles['TableCell'])
            ])
        
        table = Table(data, colWidths=[25, 155, 65, 50, 70, 120, 45])
        table.setStyle(self._get_pivot_table_style(len(data)))
        story.append(table)

    def _add_batch_metrics_analysis(self, story, results: List[AnalysisResult]):
        """Detailed metrics analysis for batch"""
        story.append(Paragraph("Metric-wise Analysis", self.styles['SectionTitle']))
        story.append(Spacer(1, 8))
        
        metric_configs = {
            'gradient': {
                'name': 'Gradient-Field PCA Analysis',
                'keys': ['eigenvalue_ratio', 'gradient_vectors_sampled'],
                'labels': ['Eigenvalue\nRatio', 'Vectors\nSampled']
            },
            'frequency': {
                'name': 'Frequency Domain Analysis (FFT)',
                'keys': ['hf_ratio', 'roughness', 'spectral_deviation'],
                'labels': ['HF Ratio', 'Roughness', 'Spec.\nDeviation']
            },
            'noise': {
                'name': 'Noise Pattern Analysis',
                'keys': ['mean_noise', 'cv', 'patches_valid'],
                'labels': ['Mean Noise', 'CV', 'Patches\nValid']
            },
            'texture': {
                'name': 'Texture Statistical Analysis',
                'keys': ['smooth_ratio', 'contrast_mean', 'entropy_mean'],
                'labels': ['Smooth\nRatio', 'Mean\nContrast', 'Mean\nEntropy']
            },
            'color': {
                'name': 'Color Distribution Analysis',
                'keys': ['saturation_stats.mean_saturation', 'saturation_stats.high_sat_ratio'],
                'labels': ['Mean\nSaturation', 'High Saturation\nRatio']
            }
        }
        
        for metric_key, config in metric_configs.items():
            story.append(Paragraph(config['name'], self.styles['SectionHeader']))
            
            # Build header
            header = [
                Paragraph("<b>#</b>", self.styles['TableHeader']),
                Paragraph("<b>Filename</b>", self.styles['TableHeader']),
                Paragraph("<b>Score</b>", self.styles['TableHeader']),
                Paragraph("<b>Confidence</b>", self.styles['TableHeader'])
            ]
            
            for label in config['labels']:
                header.append(Paragraph(f"<b>{label}</b>", self.styles['TableHeader']))
            
            data = [header]
            
            for idx, result in enumerate(results, 1):
                metric_result = result.metric_results.get(metric_key)
                if not metric_result:
                    continue
                
                details = metric_result.details or {}
                
                row = [
                    Paragraph(str(idx), self.styles['TableCellSmall']),
                    Paragraph(result.filename, self.styles['TableCellSmall']),
                    Paragraph(f"<b>{metric_result.score:.3f}</b>", self.styles['TableCellSmall']),
                    Paragraph(f"{metric_result.confidence:.2f}" if metric_result.confidence else "N/A", 
                             self.styles['TableCellSmall'])
                ]
                
                # Extract values
                for key in config['keys']:
                    value = self._extract_nested_value(details, key)
                    formatted_value = self._format_value(value, decimal_places=3)
                    row.append(Paragraph(formatted_value, self.styles['TableCellSmall']))
                
                data.append(row)
            
            # Dynamic column widths
            num_detail_cols = len(config['labels'])
            detail_col_width = (530 - 25 - 140 - 45 - 35) // num_detail_cols
            col_widths = [25, 140, 45, 35] + [detail_col_width] * num_detail_cols
            
            table = Table(data, colWidths=col_widths)
            table.setStyle(self._get_pivot_table_style(len(data)))
            story.append(table)
            story.append(Spacer(1, 10))

    def _add_individual_results_summary(self, story, results: List[AnalysisResult]):
        """Individual image summaries in batch"""
        story.append(Paragraph("Individual Image Summaries", self.styles['SectionTitle']))
        story.append(Spacer(1, 8))
        
        for idx, result in enumerate(results, 1):
            if idx > 1:
                story.append(Spacer(1, 12))
            
            story.append(Paragraph(f"Image {idx}: {result.filename}", self.styles['SectionHeader']))
            
            # Quick stats
            quick_data = [
                [Paragraph("<b>Property</b>", self.styles['TableHeader']), 
                 Paragraph("<b>Value</b>", self.styles['TableHeader'])],
                [Paragraph("Score", self.styles['TableCell']), 
                 Paragraph(f"<b>{result.overall_score:.4f}</b>", self.styles['TableCell'])],
                [Paragraph("Status", self.styles['TableCell']), 
                 Paragraph(self._get_status_badge_html(result.status.value), self.styles['TableCell'])],
                [Paragraph("Confidence", self.styles['TableCell']), 
                 Paragraph(f"{result.confidence}%", self.styles['TableCell'])],
                [Paragraph("Dimensions", self.styles['TableCell']), 
                 Paragraph(f"{result.image_size[0]} × {result.image_size[1]}", self.styles['TableCell'])],
            ]
            
            quick_table = Table(quick_data, colWidths=[120, 410])
            quick_table.setStyle(self._get_standard_table_style(len(quick_data)))
            story.append(quick_table)
            story.append(Spacer(1, 5))
            
            # Top 3 signals
            story.append(Paragraph("Top Detection Signals:", self.styles['SubHeader']))
            
            top_signals = sorted(result.signals, key=lambda s: s.score, reverse=True)[:3]
            signal_data = [[
                Paragraph("<b>Signal</b>", self.styles['TableHeader']),
                Paragraph("<b>Score</b>", self.styles['TableHeader']),
                Paragraph("<b>Status</b>", self.styles['TableHeader'])
            ]]
            
            for signal in top_signals:
                signal_data.append([
                    Paragraph(signal.name, self.styles['TableCellSmall']),
                    Paragraph(f"{signal.score:.3f}", self.styles['TableCellSmall']),
                    Paragraph(self._get_status_badge_html(signal.status.value), self.styles['TableCellSmall'])
                ])
            
            signal_table = Table(signal_data, colWidths=[200, 165, 165])
            signal_table.setStyle(self._get_standard_table_style(len(signal_data)))
            story.append(signal_table)

    def _add_batch_recommendations(self, story, batch_result: BatchAnalysisResult):
        """Batch-level recommendations"""
        story.append(Paragraph("Batch Analysis Recommendations", self.styles['SectionTitle']))
        story.append(Spacer(1, 8))
        
        review_count = batch_result.summary.get('review_required', 0)
        total = batch_result.total_images
        
        if review_count > 0:
            rec_text = f"""
            <b>⚠️ ACTION REQUIRED</b><br/>
            {review_count} out of {total} images require manual review ({review_count/total*100:.1f}%).<br/>
            <br/>
            <b>Recommended Actions:</b><br/>
            • Prioritize high-risk images for immediate review<br/>
            • Assign qualified personnel for verification<br/>
            • Document review findings and decisions<br/>
            • Consider additional forensic analysis for flagged images<br/>
            • Update screening protocols based on findings
            """
            rec_color = self.COLOR_LIGHT_ORANGE
            border_color = self.COLOR_WARNING
        else:
            rec_text = f"""
            <b>✅ BATCH PASSED SCREENING</b><br/>
            All {total} images appear to be authentic based on current analysis.<br/>
            <br/>
            <b>Recommended Actions:</b><br/>
            • Archive results for audit trail<br/>
            • Maintain periodic re-screening schedule<br/>
            • Monitor for evolving AI generation techniques<br/>
            • Update detection models regularly<br/>
            • Document chain of custody
            """
            rec_color = self.COLOR_LIGHT_GREEN
            border_color = self.COLOR_SUCCESS
        
        rec_table = Table([[Paragraph(rec_text, self.styles['CustomBodyText'])]], colWidths=[530])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), rec_color),
            ('BOX', (0, 0), (-1, -1), 2, border_color),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12)
        ]))
        story.append(rec_table)

    def _add_professional_footer(self, story):
        """Professional footer with disclaimers"""
        story.append(Spacer(1, 15))
        
        disclaimer_lines = [
            "⚠️ <b>DISCLAIMER</b>: This report provides probabilistic screening results based on current AI detection methodologies, not definitive verdicts.",
            "Results should be manually verified for critical applications. False positive rate: ~10-20%. Accuracy may vary with image quality and AI generation techniques.",
            "This analysis should be used as one component of a comprehensive verification process, not as the sole basis for decision-making.",
            "© 2025 AI Image Screener | Confidential Report | For Authorized Use Only"
        ]
        
        for line in disclaimer_lines:
            story.append(Paragraph(line, self.styles['Footer']))
            story.append(Spacer(1, 2))

    # Helper methods
    
    def _get_standard_table_style(self, num_rows):
        """Standard table styling"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLOR_ALT_ROW]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5)
        ])

    def _get_signal_table_style(self, num_rows):
        """Signal table styling with color coding"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLOR_ALT_ROW]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5)
        ])

    def _get_pivot_table_style(self, num_rows):
        """Pivot table styling"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_HEADER_BG),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLOR_ALT_ROW]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
        ])

    def _get_status_badge_html(self, status: str) -> str:
        """Generate status badge HTML"""
        if status == "REVIEW_REQUIRED" or status == "flagged":
            return "<font color='#B71C1C'><b>🔴 FLAGGED</b></font>"
        elif status == "warning":
            return "<font color='#E65100'><b>🟡 WARNING</b></font>"
        else:
            return "<font color='#1B5E20'><b>🟢 PASSED</b></font>"

    def _get_status_badge_short(self, status: str) -> str:
        """Short status badge"""
        if status == "REVIEW_REQUIRED":
            return "<font color='#B71C1C'><b>⚠️ REVIEW REQUIRED</b></font>"
        else:
            return "<font color='#1B5E20'><b>✓ LIKELY AUTHENTIC</b></font>"

    def _get_metric_status_html(self, score: float) -> str:
        """Metric status based on score"""
        if score > 0.7:
            return "<font color='#B71C1C'><b>High Risk</b></font>"
        elif score > 0.5:
            return "<font color='#E65100'><b>Moderate Risk</b></font>"
        else:
            return "<font color='#1B5E20'><b>Low Risk</b></font>"

    def _format_value(self, value: Any, decimal_places: int = 4) -> str:
        """Format value for display"""
        if value is None or (isinstance(value, dict) and 'reason' in value):
            return "N/A"
        elif isinstance(value, float):
            return f"{value:.{decimal_places}f}"
        elif isinstance(value, (int, str, bool)):
            return str(value)
        else:
            return "N/A"

    def _extract_nested_value(self, details: dict, key: str) -> Any:
        """Extract nested dictionary values"""
        if '.' in key:
            parts = key.split('.')
            value = details
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, None)
                else:
                    return None
            return value
        else:
            return details.get(key, None)