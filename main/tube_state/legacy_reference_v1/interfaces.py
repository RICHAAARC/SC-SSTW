from abc import ABC, abstractmethod
from .schema import LatentSample

class EvidenceExtractor(ABC):
    @abstractmethod
    def extract(self, sample: LatentSample):
        raise NotImplementedError
