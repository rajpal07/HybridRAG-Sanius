import re
import logging
from typing import Optional

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from backend.config import SPACY_MODEL

logger = logging.getLogger(__name__)

_analyzer: Optional[AnalyzerEngine] = None
_anonymizer: Optional[AnonymizerEngine] = None


def _create_spacy_analyzer(model_name: str) -> AnalyzerEngine:
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
    )
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


# Medical measurement patterns that spaCy wrongly classifies as DATE_TIME.
# These are used to post-filter false-positive DATE_TIME results.
_MEDICAL_FP_PATTERNS = re.compile(
    r"""
    (?:
        \b\d+(?:\.\d+)?\s*(?:yr|year|years|month|months|week|weeks|day|days)\s*old\b  # ages: 42 yr old
        | \b\d+(?:\.\d+)?\s*g/dL\b                                                    # lab values: 14.5g/dL
        | \b\d+(?:\.\d+)?\s*(?:mmol|mg|mcg|IU|mEq|mL|L|kg|cm|mm)/\S*               # any unit-qualified measurement
        | \b\d+\.\d+g\b                                                               # e.g. 14.5g (Hb shorthand)
        | \b\d+/10\b                                                                  # pain scores: 4/10, 3/10
        | \b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+
          (?:hour|hours|day|days|week|weeks|month|months|year|years)\b               # durations: one week, 2 months
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _build_analyzer() -> AnalyzerEngine:
    # Prefer the configured model, but fall back to a smaller default model to
    # keep local setup friction low.
    try:
        analyzer = _create_spacy_analyzer(SPACY_MODEL)
    except Exception as e:
        fallback = "en_core_web_sm"
        logger.warning(
            "Failed to load spaCy model '%s' (%s). Falling back to '%s'.",
            SPACY_MODEL,
            e,
            fallback,
        )
        analyzer = _create_spacy_analyzer(fallback)

    # ------------------------------------------------------------------ #
    # Strip DATE_TIME from the spaCy NER recognizer — it produces too many
    # false positives in clinical text (ages, lab values, pain scores, and
    # relative durations all get mislabelled as dates).  We replace it with
    # a strict pattern-only recognizer below.
    # ------------------------------------------------------------------ #
    for rec in analyzer.registry.recognizers:
        if rec.__class__.__name__ == "SpacyRecognizer" and "DATE_TIME" in getattr(rec, "supported_entities", []):
            rec.supported_entities = [e for e in rec.supported_entities if e != "DATE_TIME"]
            logger.debug("Removed DATE_TIME from SpacyRecognizer; will use StrictDateRecognizer instead.")
            break

    # Strict calendar-date patterns only — avoids matching ages, scores, lab
    # values, and plain durations that spaCy's NER incorrectly labels as dates.
    _MONTHS = (
        r"(?:January|February|March|April|May|June|July|August"
        r"|September|October|November|December"
        r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    )
    strict_date_patterns = [
        # ISO: 2024-05-10
        Pattern("iso_date", r"\b\d{4}-\d{2}-\d{2}\b", 0.95),
        # UK numeric: 14/10/2024 or 14-10-2024 (requires 4-digit year to avoid 4/10)
        Pattern("uk_numeric_date", r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\b", 0.90),
        # Long form: 14th October 2024 / 14 October 2024
        Pattern("long_date", rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\s+\d{{4}}\b", 0.95),
        # Month + year: October 2024
        Pattern("month_year", rf"\b{_MONTHS}\s+\d{{4}}\b", 0.85),
        # Unambiguous relative anchors only (not "one week", "two months")
        Pattern("relative_date", r"\b(?:today|yesterday|tomorrow)\b", 0.70),
    ]
    date_recognizer = PatternRecognizer(
        supported_entity="DATE_TIME",
        patterns=strict_date_patterns,
        name="StrictDateRecognizer",
    )
    analyzer.registry.add_recognizer(date_recognizer)

    # UK NHS number: 3 digits, separator, 3 digits, separator, 4 digits
    nhs_pattern = Pattern(
        name="uk_nhs_number",
        regex=r"\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b",
        score=0.9,
    )
    nhs_recognizer = PatternRecognizer(
        supported_entity="UK_NHS_CUSTOM",
        patterns=[nhs_pattern],
        name="NHSNumberRecognizer",
    )
    analyzer.registry.add_recognizer(nhs_recognizer)

    # Hospital IDs: 2-3 uppercase letters followed by 5-8 digits
    hospital_id_pattern = Pattern(
        name="hospital_id",
        regex=r"\b[A-Z]{2,3}\d{5,8}\b",
        score=0.75,
    )
    hospital_id_recognizer = PatternRecognizer(
        supported_entity="HOSPITAL_ID",
        patterns=[hospital_id_pattern],
        name="HospitalIDRecognizer",
    )
    analyzer.registry.add_recognizer(hospital_id_recognizer)

    # Clinician names: PERSON entity preceded by title within ~10 chars
    clinician_recognizer = _ClinicianRecognizer()
    analyzer.registry.add_recognizer(clinician_recognizer)

    return analyzer


class _ClinicianRecognizer(PatternRecognizer):
    TITLES = re.compile(
        r"\b(Dr\.?|Prof\.?|Professor|Nurse|Consultant|Mr\.?|Mrs\.?|Ms\.?)\s+",
        re.IGNORECASE,
    )

    def __init__(self):
        name_pattern = Pattern(
            name="clinician_name",
            regex=r"\b(Dr\.?|Prof\.?|Professor|Nurse|Consultant)\s+[A-Z][a-z]+([\s\-][A-Z][a-z]+)*\b",
            score=0.85,
        )
        super().__init__(
            supported_entity="CLINICIAN_NAME",
            patterns=[name_pattern],
            name="ClinicianRecognizer",
        )


def get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        logger.info("Loading Presidio analyzer with spaCy model '%s' …", SPACY_MODEL)
        _analyzer = _build_analyzer()
    return _analyzer


def get_anonymizer() -> AnonymizerEngine:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


ENTITY_OPERATORS = {
    "PERSON":          OperatorConfig("replace", {"new_value": "[PATIENT_NAME]"}),
    "PHONE_NUMBER":    OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "UK_NHS":          OperatorConfig("replace", {"new_value": "[NHS_NUMBER]"}),
    "UK_NHS_CUSTOM":   OperatorConfig("replace", {"new_value": "[NHS_NUMBER]"}),
    "DATE_TIME":       OperatorConfig("replace", {"new_value": "[DATE]"}),
    "LOCATION":        OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    "CLINICIAN_NAME":  OperatorConfig("replace", {"new_value": "[CLINICIAN]"}),
    "HOSPITAL_ID":     OperatorConfig("replace", {"new_value": "[HOSPITAL_ID]"}),
}

TARGET_ENTITIES = list(ENTITY_OPERATORS.keys())


def redact(text: str) -> dict:
    if not text or not text.strip():
        return {"original": text, "redacted": text, "entities_found": 0, "entity_types": []}

    try:
        analyzer = get_analyzer()
        anonymizer = get_anonymizer()

        raw_results: list[RecognizerResult] = analyzer.analyze(
            text=text,
            language="en",
            entities=TARGET_ENTITIES,
        )

        # Post-filter: drop PERSON results with score < 0.6 (catches misspelled
        # words that spaCy NER mislabels as names, e.g. "contiune" → PERSON)
        # and drop any DATE_TIME that still matches a medical measurement pattern.
        results = [
            r for r in raw_results
            if not (
                r.entity_type == "PERSON" and r.score < 0.6
            ) and not (
                r.entity_type == "DATE_TIME"
                and _MEDICAL_FP_PATTERNS.search(text[r.start:r.end])
            )
        ]
        if len(results) != len(raw_results):
            logger.debug(
                "Presidio FP filter removed %d result(s) from redaction.",
                len(raw_results) - len(results),
            )

        entity_types = list({r.entity_type for r in results})

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=ENTITY_OPERATORS,
        )

        return {
            "original": text,
            "redacted": anonymized.text,
            "entities_found": len(results),
            "entity_types": entity_types,
        }

    except Exception as e:
        logger.error(f"Redaction error: {e}")
        return {
            "original": text,
            "redacted": text,
            "entities_found": 0,
            "entity_types": [],
        }
