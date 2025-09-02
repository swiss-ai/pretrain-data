from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class CodeMetricsThresholdFilter(BaseFilter):
    name = "Code Quality Threshold Filter"

    def __init__(
        self,
        clarity_threshold: int,
        educational_threshold: int,
        practice_threshold: int,
        difficulty_threshold: int,
    ):
        super().__init__(exclusion_writer)
        self.clarity_threshold = clarity_threshold
        self.educational_threshold = educational_threshold
        self.practice_threshold = practice_threshold
        self.difficulty_threshold = difficulty_threshold

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        clarity = doc.metadata["clarity"]
        educational = doc.metadata["educational"]
        practice = doc.metadata["practice"]
        difficulty = doc.metadata["difficulty"]

        if (
            clarity < self.clarity_threshold
            or educational < self.educational_threshold
            or practice < self.practice_threshold
            or difficulty < self.difficulty_threshold
        ):
            return False, "code_quality_low"

        return True


class CodeQualityThresholdFilter(BaseFilter):
    name = "Code Quality Threshold Filter"

    def __init__(
        self,
        quality_threshold: int,
    ):
        super().__init__(exclusion_writer)
        self.quality_threshold = quality_threshold

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        quality = doc.metadata["quality"]

        if quality < self.quality_threshold:
            return False, "code_quality_low"

        return True
