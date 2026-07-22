"""
===========================================================
AgriSpectralSynth

Base Dataset Class

Author:
Juan Carlos Vega
OpenAI Collaboration

License:
MIT
===========================================================
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDataset(ABC):
    """
    Base abstract class for every dataset supported by
    AgriSpectralSynth.

    Any new dataset should inherit from this class.
    """

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

    # --------------------------------------------------

    @abstractmethod
    def download(self):
        """
        Download dataset if necessary.
        """
        pass

    # --------------------------------------------------

    @abstractmethod
    def prepare(self):
        """
        Organize folders and convert files
        to AgriSpectralSynth format.
        """
        pass

    # --------------------------------------------------

    @abstractmethod
    def load(self):
        """
        Return dataset samples.
        """
        pass

    # --------------------------------------------------

    @abstractmethod
    def statistics(self):
        """
        Print dataset statistics.
        """
        pass

    # --------------------------------------------------

    def exists(self):

        return self.root_dir.exists()

    # --------------------------------------------------

    def create_directory(self):

        self.root_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------

    def __str__(self):

        return (
            f"{self.__class__.__name__}"
            f"(root='{self.root_dir}')"
        )