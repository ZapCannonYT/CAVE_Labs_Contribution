import requests

RXNORM_URL = "https://rxnav.nlm.nih.gov/REST"
DAILYMED_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# simple in-memory cache
_DRUG_CACHE = {}


def normalize_drug_name(name: str):
    try:
        res = requests.get(f"{RXNORM_URL}/rxcui.json", params={"name": name})
        data = res.json()
        ids = data.get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None
    except:
        return None


def get_drug_label(rxcui: str):
    try:
        res = requests.get(f"{DAILYMED_URL}/spls.json", params={"rxcui": rxcui})
        data = res.json()

        if not data.get("data"):
            return None

        return data["data"][0]  # first match
    except:
        return None


def get_drug_info(name: str):
    name_key = name.lower()

    if name_key in _DRUG_CACHE:
        return _DRUG_CACHE[name_key]

    rxcui = normalize_drug_name(name)

    if not rxcui:
        return None

    label = get_drug_label(rxcui)

    info = {
        "name": name,
        "rxcui": rxcui,
        "label": label,
    }

    _DRUG_CACHE[name_key] = info
    return info