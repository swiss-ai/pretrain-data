from data_pipeline_pretrain.utils import list_files
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from functools import lru_cache
from urllib.parse import urlparse
from protego import Protego

from typing import Set

_DEFAULT_REMOVE_USER_AGENTS = [
    "AI2Bot",  # AI2
    "Applebot-Extended",  # Apple
    "Bytespider",  # Bytedance
    "CCBot",  # Common Crawl
    "CCBot/2.0",  # Common Crawl
    "CCBot/1.0",  # Common Crawl
    "ClaudeBot",  # Anthropic
    "cohere-training-data-crawler",  # Cohere
    "Diffbot",  # Diffbot
    "FacebookBot",  # Meta
    "Meta-ExternalAgent",  # Meta
    "Google-Extended",  # Google
    "GPTBot",  # OpenAI
    "PanguBot",  # Huawei
    "*",
]


class RobotsTxtFilter(BaseFilter):
    name = "Robots.txt Filter"

    def __init__(
        self,
        robots_dict: dict[str, str],
        remove_user_agents: set[str] = _DEFAULT_REMOVE_USER_AGENTS,
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(exclusion_writer)
        self.robots_dict = robots_dict
        self.remove_user_agents = remove_user_agents

    @lru_cache(maxsize=8192)
    def _get_parser(self, domain: str):
        robots_txt_content = self.robots_dict.get(domain, None)

        if robots_txt_content is None:
            return None
        try:
            if isinstance(robots_txt_content, bytes):
                robots_txt_content = robots_txt_content.decode(
                    "utf-8", errors="replace"
                )
            parser = Protego.parse(robots_txt_content)
            return parser
        except Exception:
            # Ignore parsing errors
            return None

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """
        Args:
            doc: Check if the documnet ID needs to be filtered, TRUE == KEPT

        Returns: TRUE if the doc.id should be KEPT, FALSE if it should be REMOVED.
        """

        url = doc.metadata["url"]
        domain = urlparse(url).netloc if url else None
        parser = self._get_parser(domain)

        # If no parser -> permissive
        if parser is None:
            return True

        disallowed_user_agents = []

        for user_agent in self.remove_user_agents:
            try:
                if not parser.can_fetch(url, user_agent):
                    disallowed_user_agents.append(user_agent)
            except Exception:
                # Error checking url -> permissive
                pass

        # If there are disallowed user agents -> not permissive
        if len(disallowed_user_agents) > 0:
            doc.metadata["disallowed_user_agents"] = disallowed_user_agents
            return False

        # Otherwise permissive
        return True


class IdFilter(BaseFilter):
    name = "Id Filter"

    def __init__(
        self,
        ids_to_filter: set[str],
        exclusion_writer: DiskWriter = None,
    ):
        """
        An example filter to remove text above or below a given threshold (removes text with less than `min_text_length` or more than `max_text_length` characters).

        Args:
            min_text_length: int
            max_text_length: int
            exclusion_writer: DiskWriter
        """
        super().__init__(exclusion_writer)
        self.ids_to_filter = ids_to_filter

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """
        Args:
            doc: Check if the documnet ID needs to be filtered, TRUE == KEPT

        Returns: TRUE if the doc.id should be KEPT, FALSE if it should be REMOVED.
        """
        return doc.id not in self.ids_to_filter


def load_robots(
    folder_path,
):
    import pandas as pd

    robots = set()
    files = list_files(f"{folder_path}")

    def load_robot(file):
        pf = pd.read_parquet(file, columns=["id", "user_agents"])
        # If user_agents is non-empty, the doc needs to be removed
        pf = pf[pf["user_agents"].map(len) > 0]
        r = pf["id"].to_list()
        return r

    for file in files:
        robot = load_robot(file)
        robots.update(robot)

    return robots
