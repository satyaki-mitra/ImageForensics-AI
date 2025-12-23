# Dependencies
import csv
from pathlib import Path
from typing import Optional
from datetime import datetime
from utils.logger import get_logger
from config.settings import settings
from config.constants import MetricType
from config.schemas import AnalysisResult
from config.constants import FinalDecision
from utils.helpers import generate_unique_id
from config.schemas import BatchAnalysisResult
from features.detailed_result_maker import DetailedResultMaker


# Setup Logging
logger = get_logger(__name__)


class CSVReporter:
    """
    CSV report generator 

    Guarantees:
    -----------
    - FinalDecision is authoritative
    - Metrics are informational only
    - Evidence-first reporting
    - Audit-safe CSV structure
    """
    def __init__(self):
        """
        Initialize CSV Reporter
        """
        self.detailed_maker = DetailedResultMaker()

        logger.debug("CSVReporter initialized")


    def export_batch_summary(self, batch_result: BatchAnalysisResult, output_dir: Optional[Path] = None) -> Path:
        """
        Export batch decision summary as CSV
        """
        output_dir  = output_dir or settings.REPORTS_DIR
        report_id   = generate_unique_id()
        filename    = f"batch_summary_{report_id}.csv"
        output_path = output_dir / filename

        logger.info(f"Generating batch summary CSV: {filename}")

        try:
            with open(output_path, 'w', newline = '', encoding = 'utf-8-sig') as f:
                writer = csv.writer(f)

                self._write_report_header(writer,
                                          report_type = "Batch Decision Summary",
                                          timestamp   = batch_result.timestamp,
                                         )

                self._write_batch_decision_statistics(writer       = writer, 
                                                      batch_result = batch_result,
                                                     )

                self._write_batch_results_table(writer       = writer,
                                                batch_result = batch_result,
                                               )

                self._write_footer(writer = writer)

            logger.info(f"Batch summary CSV generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate batch summary CSV: {e}")
            raise


    def export_batch_detailed(self, batch_result: BatchAnalysisResult, output_dir: Optional[Path] = None) -> Path:
        """
        Export detailed batch forensic CSV
        """
        output_dir  = output_dir or settings.REPORTS_DIR
        report_id   = generate_unique_id()
        filename    = f"batch_detailed_{report_id}.csv"
        output_path = output_dir / filename

        logger.info(f"Generating detailed batch CSV: {filename}")

        try:
            with open(output_path, 'w', newline = '', encoding = 'utf-8-sig') as f:
                writer = csv.writer(f)

                self._write_report_header(writer,
                                          report_type = "Detailed Batch Analysis",
                                          timestamp   = batch_result.timestamp,
                                         )

                for idx, result in enumerate(batch_result.results, 1):
                    self._write_detailed_image_section(writer,
                                                       result       = result,
                                                       image_number = idx,
                                                       total_images = batch_result.processed,
                                                      )

                    if (idx < batch_result.processed):
                        writer.writerow([])
                        writer.writerow(['=' * 100])
                        writer.writerow([])

                self._write_footer(writer = writer)

            logger.info(f"Detailed batch CSV generated: {output_path}")
            
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate detailed batch CSV: {e}")
            raise


    def export_single_detailed(self, result: AnalysisResult, output_dir: Optional[Path] = None) -> Path:
        """
        Export single image detailed CSV
        """
        output_dir  = output_dir or settings.REPORTS_DIR
        report_id   = generate_unique_id()
        filename    = f"single_analysis_{report_id}.csv"
        output_path = output_dir / filename

        logger.info(f"Generating single image CSV: {filename}")

        try:
            with open(output_path, 'w', newline = '', encoding = 'utf-8-sig') as f:
                writer = csv.writer(f)

                self._write_report_header(writer,
                                          report_type = "Single Image Analysis",
                                          timestamp   = result.timestamp,
                                         )

                self._write_detailed_image_section(writer,
                                                   result       = result,
                                                   image_number = 1,
                                                   total_images = 1,
                                                  )

                self._write_footer(writer = writer)

            logger.info(f"Single image CSV generated: {output_path}")
            
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate single image CSV: {e}")
            raise


    def _write_report_header(self, writer, report_type: str, timestamp: datetime) -> None:
        writer.writerow(['=' * 100])
        writer.writerow([f'AI Image Screener - {report_type}'])
        writer.writerow([f'Generated: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([f'Version: {settings.VERSION}'])
        writer.writerow(['=' * 100])
        writer.writerow([])


    def _write_batch_decision_statistics(self, writer, batch_result: BatchAnalysisResult) -> None:
        writer.writerow(['BATCH DECISION STATISTICS'])
        writer.writerow([])

        summary = batch_result.summary or {}

        rows    = [['Total Images', batch_result.total_images],
                   ['Processed', batch_result.processed],
                   ['Failed', batch_result.failed],
                   ['Success Rate', f"{summary.get('success_rate', 0)}%"],
                   ['', ''],
                  ]

        for decision in FinalDecision:
            rows.append([decision.value, summary.get(decision.value, 0)])

        rows.append(['Total Processing Time', f"{batch_result.total_processing_time:.2f}s"])

        for row in rows:
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(['=' * 100])
        writer.writerow([])


    def _write_batch_results_table(self, writer, batch_result: BatchAnalysisResult) -> None:
        writer.writerow(['ANALYSIS RESULTS'])
        writer.writerow([])

        header = ['Filename',
                  'Final Decision',
                  'Decision Confidence (%)',
                  'Overall Score (informational)',
                  'Decision Explanation',
                  'Processing Time (s)',
                 ]

        writer.writerow(header)

        for result in batch_result.results:
            writer.writerow([result.filename,
                             result.final_decision.value,
                             f"{result.confidence}%",
                             f"{result.overall_score:.3f}",
                             (result.decision_explanation or '').replace("\n", " "),
                             f"{result.processing_time:.2f}",
                           ])

        writer.writerow([])


    def _write_detailed_image_section(self, writer, result: AnalysisResult, image_number: int, total_images: int) -> None:
        writer.writerow([f'IMAGE {image_number} OF {total_images}'])
        writer.writerow([])

        # Decision Summary
        writer.writerow(['FINAL DECISION'])
        writer.writerow(['Decision', result.final_decision.value])
        writer.writerow(['Confidence', f"{result.confidence}%"])
        writer.writerow(['Explanation', result.decision_explanation or ''])
        writer.writerow([])

        # Evidence Summary
        if result.evidence:
            writer.writerow(['EVIDENCE SUMMARY'])
            writer.writerow(['Source', 'Direction', 'Strength', 'Confidence', 'Finding'])

            for e in result.evidence:
                writer.writerow([e.source.value,
                                 e.direction.value,
                                 e.strength.value,
                                 f"{e.confidence:.3f}" if e.confidence is not None else 'N/A',
                                 e.finding.replace("\n", " "),
                               ])

            writer.writerow([])

        # Metric Signals (Informational)
        writer.writerow(['METRIC SIGNALS (INFORMATIONAL)'])
        writer.writerow(['Metric', 'Score', 'Status', 'Confidence'])

        for signal in result.signals:
            metric_result = result.metric_results.get(signal.metric_type)

            writer.writerow([signal.name,
                             f"{signal.score:.3f}",
                             signal.status.value,
                             f"{metric_result.confidence:.3f}" if (metric_result and metric_result.confidence is not None) else 'N/A',
                           ])

        writer.writerow([])


    def _write_footer(self, writer) -> None:
        writer.writerow(['=' * 100])
        writer.writerow(['Report generated by AI Image Screener'])
        writer.writerow(['DISCLAIMER: Statistical signals are non-decisional'])
        writer.writerow(['Final decisions are policy-based and auditable'])
        writer.writerow(['=' * 100])
