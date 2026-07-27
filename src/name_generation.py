"""Nationality-aware synthetic participant metadata helpers."""

from __future__ import annotations

import calendar
import random
from datetime import date
from typing import Any


NAME_POOLS: dict[str, dict[str, list[str]]] = {
    "Greece": {
        "first_names": [
            "Nikos",
            "Maria",
            "Dimitris",
            "Eleni",
            "Anna",
            "George",
            "Sofia",
            "Panagiotis",
            "Katerina",
            "Michalis",
        ],
        "last_names": [
            "Papadopoulos",
            "Papadopoulou",
            "Georgiou",
            "Nikolaou",
            "Manolis",
            "Papastathis",
            "Stavropoulos",
            "Tsoukalas",
        ],
    },
    "Cyprus": {
        "first_names": [
            "Andreas",
            "Despina",
            "Elena",
            "Costas",
            "Maria",
            "Michalis",
            "Yiannis",
            "Katerina",
        ],
        "last_names": [
            "Constantinou",
            "Demetriou",
            "Ioannou",
            "Pavlou",
            "Georgiou",
            "Skepsi",
        ],
    },
    "Bulgaria": {
        "first_names": [
            "Ivan",
            "Elena",
            "Nikolay",
            "Maria",
            "Dimitar",
            "Krasimira",
            "Petar",
            "Veselina",
        ],
        "last_names": [
            "Petrov",
            "Ivanova",
            "Dimitrov",
            "Koleva",
            "Stoyanov",
            "Hristova",
            "Vasilev",
        ],
    },
    "Romania": {
        "first_names": [
            "Andrei",
            "Elena",
            "Mihai",
            "Ioana",
            "Ana",
            "Vlad",
            "Cristina",
            "Bogdan",
        ],
        "last_names": [
            "Popescu",
            "Ionescu",
            "Stan",
            "Dumitrescu",
            "Marin",
            "Radu",
            "Nistor",
        ],
    },
    "Albania": {
        "first_names": [
            "Arben",
            "Elira",
            "Blerim",
            "Arta",
            "Gentian",
            "Ilir",
            "Era",
            "Erion",
        ],
        "last_names": [
            "Hoxha",
            "Kodra",
            "Berisha",
            "Leka",
            "Dervishi",
            "Gashi",
            "Krasniqi",
        ],
    },
}

DEFAULT_NAME_POOL = {
    "first_names": ["Maria", "Nikos", "Elena", "Andrei", "Arben"],
    "last_names": ["Papadopoulos", "Constantinou", "Petrov", "Popescu", "Hoxha"],
}

BACKGROUND_OPTIONS = [
    "Engineer",
    "Teacher",
    "Doctor",
    "Manager",
    "Student",
    "Consultant",
    "Developer",
    "Designer",
    "Sales",
    "HR",
]


def _random_birth_date(age: int, rng: random.Random = random) -> str:
    """Generate a birth date string that roughly matches the given age."""
    current_year = date.today().year
    year = max(1900, current_year - age)
    month = rng.randint(1, 12)
    day = rng.randint(1, calendar.monthrange(year, month)[1])
    return f"{day:02d}/{month:02d}/{year}"


def _pick_name_pool(nationality: str | None) -> dict[str, list[str]]:
    return NAME_POOLS.get(nationality or "", DEFAULT_NAME_POOL)


def generate_participant_metadata(
    nationality: str,
    *,
    rng: random.Random = random,
    age: int | None = None,
    background: str | None = None,
) -> dict[str, Any]:
    """Create a synthetic participant profile that matches the nationality."""
    pool = _pick_name_pool(nationality)
    participant_age = age if age is not None else rng.randint(25, 70)

    return {
        "FirstName": rng.choice(pool["first_names"]),
        "LastName": rng.choice(pool["last_names"]),
        "Age": participant_age,
        "DateOfBirth": _random_birth_date(participant_age, rng=rng),
        "Nationality": nationality,
        "Background": background if background is not None else rng.choice(BACKGROUND_OPTIONS),
    }
