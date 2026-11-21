"""Phase 4: Curated Taxonomy Skill Extractor."""

import json
import re
from pathlib import Path
from config import TAXONOMY_PATH


class SkillExtractor:
    def __init__(self, taxonomy_path: Path = TAXONOMY_PATH):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy: dict[str, dict[str, list[str]]] = json.load(f)

        self._compiled_patterns: list[tuple[str, str, re.Pattern]] = []
        for category, skills_map in self.taxonomy.items():
            for canonical_skill, aliases in skills_map.items():
                escaped = [re.escape(alias.lower()) for alias in aliases]
                pattern_str = r"(?<![a-z0-9])(?:" + "|".join(escaped) + r")(?![a-z0-9])"
                self._compiled_patterns.append(
                    (category, canonical_skill, re.compile(pattern_str, re.IGNORECASE))
                )

    def extract_skills(self, text: str) -> dict[str, list[str]]:
        categorized: dict[str, list[str]] = {cat: [] for cat in self.taxonomy.keys()}
        technical_skills: list[str] = []
        soft_skills: list[str] = []

        for category, canonical_skill, pattern in self._compiled_patterns:
            if pattern.search(text):
                if canonical_skill not in categorized[category]:
                    categorized[category].append(canonical_skill)
                if category == "soft_skills":
                    if canonical_skill not in soft_skills:
                        soft_skills.append(canonical_skill)
                else:
                    if canonical_skill not in technical_skills:
                        technical_skills.append(canonical_skill)

        return {
            "by_category": categorized,
            "technical_skills": sorted(technical_skills),
            "soft_skills": sorted(soft_skills),
            "all_skills": sorted(set(technical_skills + soft_skills)),
        }
