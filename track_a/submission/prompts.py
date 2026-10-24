"""Label JSON -> text prompt that restates the features the evaluator checks."""

GENERIC = "a photo of the interior of a car, realistic, natural light"


def _person(role: str, label: dict, prefix: str, belt_key: str) -> str | None:
    gender = (label.get(f"{prefix}_gender") or "").lower()
    color = (label.get(f"{prefix}_tshirt_color") or "").lower()
    emotion = (label.get(f"{prefix}_emotion") or "").lower()
    belt = label.get(belt_key)
    parts = [f"a {emotion} " if emotion else "a ", f"{gender} person " if gender else "person "]
    text = "".join(parts) + f"in the {role}"
    if color:
        text += f" wearing a {color} t-shirt"
    if belt == "YES":
        text += " with the seat belt fastened across the chest"
    elif belt == "NO":
        text += " without a seat belt"
    return text


def label_prompt(label: dict) -> str:
    people = []
    d = _person("driver seat", label, "driver", "driver_safety_belt")
    people.append(d)
    if label.get("driver_phone") == "YES":
        people[0] += ", holding a phone to the ear"
    if label.get("passenger_codriver") == "YES":
        people.append(_person("front passenger seat", label, "passenger_codriver", "codriver_safety_belt"))
    if label.get("passenger_rear_seat_left") == "YES":
        people.append(_person("rear left seat", label, "passenger_rear_left", "passenger_rear_left_safety_belt"))
    if label.get("passenger_rear_seat_right") == "YES":
        people.append(_person("rear right seat", label, "passenger_rear_right", "passenger_rear_right_safety_belt"))

    objs = []
    if label.get("phone_codriver_seat") == "YES":
        c = (label.get("phone_codriver_seat_color") or "").lower()
        objs.append(f"a {c} phone".replace("a  ", "a "))
    if label.get("colabottle_codriver_seat") == "YES":
        objs.append("a cola bottle")
    if label.get("colacan_codriver_seat") == "YES":
        objs.append("a cola can")
    if label.get("suitcase") == "YES":
        c = (label.get("suitcase_color") or "").lower()
        objs.append(f"a {c} suitcase".replace("a  ", "a "))
    if label.get("baby_seat") == "YES":
        objs.append("a baby seat")

    text = "A realistic photo taken inside a car. " + "; ".join(people) + "."
    if objs:
        text += " On the front passenger seat: " + ", ".join(objs) + "."
    return text + " Natural lighting, photorealistic."
