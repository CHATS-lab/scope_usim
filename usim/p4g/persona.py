"""Persona loading for Persuasion for Good.

Loads speaker persona data from convokit Corpus and formats it
as bullet-point text for prompt injection.

Ported from persuasion_simulation/src/prompts/persona.py (bullet-point format only).
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PersonaLoader:
    """Loads persona data from a convokit Corpus.

    Caches the Corpus instance so it is loaded only once.
    """

    def __init__(self, corpus_path: str):
        """Initialize with path to convokit corpus.

        Args:
            corpus_path: Path to the convokit Corpus directory
        """
        self.corpus_path = corpus_path
        self._corpus = None

    def _load_corpus(self):
        """Load the convokit Corpus (lazy, cached)."""
        if self._corpus is None:
            from convokit import Corpus

            self._corpus = Corpus(filename=self.corpus_path)
            logger.info(f"Loaded convokit corpus from {self.corpus_path}")
        return self._corpus

    def get_persona_text(self, speaker_id: str) -> str:
        """Get formatted persona text for a speaker.

        Args:
            speaker_id: Speaker ID in the corpus

        Returns:
            Persona text wrapped in <persona> tags with bullet points
        """
        corpus = self._load_corpus()
        persona = corpus.get_speaker(speaker_id)

        demographics = {
            "Age": persona.meta.get("age", ""),
            "Sex": persona.meta.get("sex", ""),
            "Race": persona.meta.get("race", ""),
            "Education": persona.meta.get("edu", ""),
            "Marital Status": persona.meta.get("marital", ""),
            "Employment": persona.meta.get("employment", ""),
            "Income": persona.meta.get("income", ""),
            "Religion": persona.meta.get("religion", ""),
            "Ideology": persona.meta.get("ideology", ""),
        }

        personality = {
            "Extroversion": persona.meta.get("extrovert", ""),
            "Agreeableness": persona.meta.get("agreeable", ""),
            "Conscientiousness": persona.meta.get("conscientious", ""),
            "Neuroticism": persona.meta.get("neurotic", ""),
            "Openness": persona.meta.get("open", ""),
        }

        morals = {
            "Care": persona.meta.get("care", ""),
            "Fairness": persona.meta.get("fairness", ""),
            "Loyalty": persona.meta.get("loyalty", ""),
            "Authority": persona.meta.get("authority", ""),
            "Purity": persona.meta.get("purity", ""),
            "Freedom": persona.meta.get("freedom", ""),
        }

        values = {
            "Conformity": persona.meta.get("conform", ""),
            "Tradition": persona.meta.get("tradition", ""),
            "Benevolence": persona.meta.get("benevolence", ""),
            "Universalism": persona.meta.get("universalism", ""),
            "Self Direction": persona.meta.get("self_direction", ""),
            "Stimulation": persona.meta.get("stimulation", ""),
            "Hedonism": persona.meta.get("hedonism", ""),
            "Achievement": persona.meta.get("achievement", ""),
            "Power": persona.meta.get("power", ""),
            "Security": persona.meta.get("security", ""),
        }

        decision = {
            "Rational": persona.meta.get("rational", ""),
            "Intuitive": persona.meta.get("intuitive", ""),
        }

        def _fmt(d: Dict[str, Any]) -> str:
            return ", ".join(f"{k}: {v}" for k, v in d.items())

        return (
            "<persona>\n"
            f"- Demographics: {_fmt(demographics)}\n"
            f"- Big-Five Personality Traits [1-5]: {_fmt(personality)}\n"
            f"- Moral Foundations [1-6]: {_fmt(morals)}\n"
            f"- Schwartz Values [1-6]: {_fmt(values)}\n"
            f"- Decision Making Style [1-5]: {_fmt(decision)}\n"
            "</persona>"
        )
