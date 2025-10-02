def get_advice(injury_class, confidence):
    advice_dict = {
        "bleeding": "Apply pressure to the wound and seek help if bleeding doesn't stop.",
        "fracture": "Immobilize the area and get medical attention immediately.",
        "burn": "Cool the burn under running water and cover with a clean cloth.",
        "allergic reaction": "Take antihistamine if available and seek help.",
        "choking": "Perform Heimlich maneuver if trained, call emergency services.",
        "dog bite": "Wash bite, apply antiseptic, seek medical care.",
        "electrical injuries": "Turn off power and call emergency services.",
        "cat claw": "Clean wound and apply antiseptic cream.",
        "mosquito bite": "Apply anti-itch cream.",
        "normal": "No visible injury. Monitor condition.",
        "frostbite": (
            "Move the person to a warm environment immediately. "
            "Warm the affected area gradually using warm (not hot) water or body heat. "
            "Do not rub the skin or use direct heat (fire, stove, heating pad). "
            "Seek medical attention as soon as possible."
        )
    }

    if confidence < 0.30:
        return "⚠️ Low confidence. Please seek professional medical help."
    return advice_dict.get(injury_class, "No advice available for this injury.")
