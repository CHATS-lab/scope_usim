"""User persona definitions for simulation variety."""

from typing import Any, Dict, Optional


# Collection of user personas for variety in training
USER_PERSONAS: Dict[str, Dict[str, Any]] = {
    "impatient_professional": {
        "name": "Alex Chen",
        "occupation": "Software Engineer",
        "personality": "Direct, time-conscious, values efficiency",
        "communication_style": "Brief messages, expects quick resolution",
        "patience_level": "low",
        "tech_savvy": True,
    },
    "friendly_retiree": {
        "name": "Margaret Wilson",
        "occupation": "Retired Teacher",
        "personality": "Patient, friendly, appreciates detailed explanations",
        "communication_style": "Conversational, may share personal stories",
        "patience_level": "high",
        "tech_savvy": False,
    },
    "frustrated_parent": {
        "name": "Jordan Martinez",
        "occupation": "Working Parent",
        "personality": "Stressed, dealing with multiple responsibilities",
        "communication_style": "May be interrupted, needs clear steps",
        "patience_level": "medium",
        "tech_savvy": True,
    },
    "detail_oriented": {
        "name": "Dr. Sarah Kim",
        "occupation": "Researcher",
        "personality": "Analytical, wants to understand the 'why'",
        "communication_style": "Asks many questions, documents everything",
        "patience_level": "high",
        "tech_savvy": True,
    },
    "elderly_confused": {
        "name": "Robert Thompson",
        "occupation": "Retired",
        "personality": "Kind but easily confused by technical terms",
        "communication_style": "May repeat questions, needs simple language",
        "patience_level": "high",
        "tech_savvy": False,
    },
    "business_owner": {
        "name": "Priya Patel",
        "occupation": "Small Business Owner",
        "personality": "Results-focused, cost-conscious",
        "communication_style": "Professional, focused on bottom line",
        "patience_level": "medium",
        "tech_savvy": True,
    },
}


def get_persona(
    name: str,
    custom_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get a persona by name with optional customization.

    Args:
        name: Persona name from USER_PERSONAS
        custom_fields: Optional fields to override

    Returns:
        Persona dictionary

    Raises:
        KeyError: If persona not found
    """
    if name not in USER_PERSONAS:
        available = list(USER_PERSONAS.keys())
        raise KeyError(f"Persona '{name}' not found. Available: {available}")

    persona = USER_PERSONAS[name].copy()
    if custom_fields:
        persona.update(custom_fields)

    return persona


def format_persona_for_prompt(persona: Dict[str, Any]) -> str:
    """Format a persona dict into prompt-friendly text.

    Args:
        persona: Persona dictionary

    Returns:
        Formatted persona description
    """
    lines = [
        f"Name: {persona.get('name', 'Anonymous')}",
        f"Occupation: {persona.get('occupation', 'Unknown')}",
        f"Personality: {persona.get('personality', 'Neutral')}",
        f"Communication Style: {persona.get('communication_style', 'Standard')}",
    ]

    if persona.get("tech_savvy") is not None:
        tech_level = "comfortable with technology" if persona["tech_savvy"] else "not very tech-savvy"
        lines.append(f"Technical ability: {tech_level}")

    return "\n".join(lines)


def get_random_persona() -> Dict[str, Any]:
    """Get a random persona from available options.

    Returns:
        Random persona dictionary
    """
    import random
    name = random.choice(list(USER_PERSONAS.keys()))
    return get_persona(name)
