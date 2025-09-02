import pandas as pd
from data_pipeline_pretrain.utils import ASSETS_PATH
import re
from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.data import DocumentsPipeline, Document
from datatrove.utils.typeshelper import StatHints


class PIIFormatter(BaseFormatter):
    """
    Detects email addresses, IP addresses, and EU-specific identifiers in the document text,
    replaces them with replacement text from the resource file, and optionally records
    the number of unique PII hits in the document metadata.
    """

    name = "PII"

    def __init__(
        self,
        remove_emails: bool = True,
        remove_ips: bool = True,
        remove_ibans: bool = True,
        remove_eu: bool = False,
        only_remove_public_ips: bool = True,
        add_pii_list_to_metadata: bool = False,  # Option to add PII list to metadata
        priorities_to_keep: list[str] = ["P0", "P1"],  # Parameter for priorities
        eu_file_path: str = f"{ASSETS_PATH}/pii/eu_regex.xlsx",  # Default file for EU regex
    ):
        super().__init__()
        self.remove_emails = remove_emails
        self.remove_ips = remove_ips
        self.remove_ibans = remove_ibans
        self.remove_eu = remove_eu
        self.add_pii_list_to_metadata = add_pii_list_to_metadata
        self.priorities_to_keep = priorities_to_keep

        self.detected_pii_set = set()  # To track detected PII and avoid duplicates

        # Load EU-specific regexes and replacements based on the priorities
        df = pd.read_excel(eu_file_path)
        priority_order = pd.Categorical(
            df["Priority"], categories=self.priorities_to_keep, ordered=True
        )
        df["Priority"] = priority_order
        df = df.sort_values("Priority").reset_index(drop=True)

        whitespace_before = r"\b"
        whitespace_after = r"(\.|$|\,|\s)"

        # Create dictionary of regex patterns and replacements
        self.eu_replacers = []
        for _, row in df.iterrows():
            priority = row["Priority"]
            if priority in self.priorities_to_keep:
                regex = whitespace_before + row["Regex"] + whitespace_after
                replacement = (
                    row["Replacement"] if "Replacement" in row else "[REDACTED]"
                )
                replacer = re.compile(regex)
                self.eu_replacers.append((priority, replacer, replacement))

        # Email and IP detection regexes
        self.email_regex = re.compile(
            r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
            r"(?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[A-Za-z0-9-]*[A-Za-z0-9]:)])"
        )
        self.ip_regex = re.compile(
            r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        )
        self.iban_regex = re.compile(r"[A-Z]{2}[0-9]{2}(?: [0-9]{4}){4} [A-Z0-9]{1,2}")

    def format(self, doc: Document) -> str:
        """
        Process the document text to detect PII, replace matches with specified tokens,
        and optionally update the document metadata with the number of unique PII hits.
        """
        text = doc.text
        self.detected_pii_set = set()  # Reset the set for each document

        # EU-specific PII detection and replacement
        if self.remove_eu:
            for priority, eu_replacer, replacement in self.eu_replacers:
                # Find all matches for the current pattern
                matches = re.finditer(eu_replacer, text)
                for match in matches:
                    pii_candidate = match.group(0)
                    # Add to PII list if not already added
                    if pii_candidate not in self.detected_pii_set:
                        self.detected_pii_set.add(pii_candidate)
                # Replace all instances of the pattern in the text
                text = re.sub(eu_replacer, replacement, text)

        # Email detection and replacement
        if self.remove_emails:
            matches = self.email_regex.findall(text)
            for email in matches:
                if email not in self.detected_pii_set:
                    self.detected_pii_set.add(email)
                    text = text.replace(email, "<email-pii>")

        # IP detection and replacement
        if self.remove_ips:
            matches = self.ip_regex.findall(text)
            for ip in matches:
                if ip not in self.detected_pii_set:
                    self.detected_pii_set.add(ip)
                    text = text.replace(ip, "<ip-pii>")

        if self.remove_ibans:
            matches = self.iban_regex.findall(text)
            for iban in matches:
                if iban not in self.detected_pii_set:
                    self.detected_pii_set.add(iban)
                    text = text.replace(iban, "<iban-pii>")

        # Update metadata with PII count and optionally with list of PII elements
        doc.metadata["pii_count"] = len(self.detected_pii_set)
        if self.add_pii_list_to_metadata:
            doc.metadata["pii_list"] = list(self.detected_pii_set)

        return text

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """
        Process each document in the pipeline, detect PII and replace it with tokens,
        and optionally update the document's metadata with the number of unique PII hits.
        """
        for doc in data:
            self.stat_update(StatHints.total)
            with self.track_time():
                doc.text = self.format(doc)
            yield doc
