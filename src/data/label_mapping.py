"""
Maps both dataset-specific label schemas to the unified 7-entity set:
  PERSON | EMAIL | PHONE | ADDRESS | URL | ID | USERNAME
"""
from __future__ import annotations

UNIFIED_LABELS = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-EMAIL",  "I-EMAIL",
    "B-PHONE",  "I-PHONE",
    "B-ADDRESS","I-ADDRESS",
    "B-URL",    "I-URL",
    "B-ID",     "I-ID",
    "B-USERNAME","I-USERNAME",
]

LABEL2ID = {l: i for i, l in enumerate(UNIFIED_LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(UNIFIED_LABELS)}

# ── Kaggle entity-type → unified ────────────────────────────────────────────
_KAGGLE: dict[str, str | None] = {
    "NAME_STUDENT": "PERSON",
    "EMAIL":        "EMAIL",
    "PHONE_NUM":    "PHONE",
    "STREET_ADDRESS": "ADDRESS",
    "URL_PERSONAL": "URL",
    "ID_NUM":       "ID",
    "USERNAME":     "USERNAME",
}

# ── ai4privacy entity-type → unified ────────────────────────────────────────
_AI4PRIVACY: dict[str, str | None] = {
    # Person
    "GIVENNAME": "PERSON", "GIVENNAME1": "PERSON", "GIVENNAME2": "PERSON",
    "SURNAME":   "PERSON", "SURNAME1":   "PERSON", "SURNAME2":   "PERSON",
    "FIRSTNAME": "PERSON", "LASTNAME":   "PERSON",
    "FULLNAME":  "PERSON", "NAME":       "PERSON", "MIDDLENAME": "PERSON",
    "PREFIX":    "PERSON", "SUFFIX":     "PERSON",
    "NICKNAME":  "PERSON", "ALIAS":      "PERSON",
    # Email
    "EMAIL":        "EMAIL", "EMAILADDRESS": "EMAIL",
    # Phone
    "PHONE":         "PHONE", "PHONENUMBER":  "PHONE",
    "TELEPHONENUM":  "PHONE", "PHONENUMBER2": "PHONE",
    "MOBILENUMBER":  "PHONE", "FAXNUMBER":    "PHONE",
    # Address
    "STREET":          "ADDRESS", "STREETADDRESS":    "ADDRESS",
    "BUILDINGNUMBER":  "ADDRESS", "CITY":             "ADDRESS",
    "STATE":           "ADDRESS", "COUNTY":           "ADDRESS",
    "COUNTRY":         "ADDRESS", "ZIPCODE":          "ADDRESS",
    "ZIP":             "ADDRESS", "POSTCODE":         "ADDRESS",
    "SECONDARYADDRESS":"ADDRESS",
    # URL / IP
    "URL":      "URL", "WEBSITE":    "URL", "DOMAIN": "URL",
    "DOMAINNAME":"URL","IPV4":       "URL", "IPV6":   "URL",
    "IPADDRESS": "URL",
    # ID documents
    "IDCARD":    "ID", "SSN":              "ID", "PASSPORT":   "ID",
    "DRIVERSLICENSE": "ID", "SOCIALINSURANCE": "ID",
    "TAXID":     "ID", "NIN":              "ID",
    "VEHICLEREGISTRATIONNUMBER": "ID",
    "CREDITCARDNUMBER": "ID", "IBAN": "ID", "ACCOUNTNUMBER": "ID",
    "EMPLOYEEID": "ID", "MEDICALRECORDNUMBER": "ID",
    "US_SSN": "ID", "US_ITIN": "ID", "US_PASSPORT": "ID",
    # Username
    "USERNAME":    "USERNAME", "ACCOUNTNAME": "USERNAME",
    "USERHANDLE":  "USERNAME", "DISPLAYNAME": "USERNAME",
    # Entities not in our schema → O
    "JOBAREA": None, "JOBTITLE": None, "JOBDESCRIPTOR": None,
    "GENDER": None, "AGE": None,
    "DOB": None, "DATEDOB": None, "BIRTHDATE": None, "BIRTHDAY": None,
    "CURRENCY": None, "CURRENCYNAME": None, "CURRENCYCODE": None,
    "AMOUNT": None, "BITCOINADDRESS": None, "ETHEREUMADDRESS": None,
    "LITECOINADDRESS": None,
    "COMPANYNAME": None, "ORGANIZATIONNAME": None,
    "DATE": None, "TIME": None,
}

_MAPS = {"kaggle": _KAGGLE, "ai4privacy": _AI4PRIVACY}

_UNKNOWN_WARNED: set[str] = set()


def map_label(label: str, source: str) -> str:
    """Convert a dataset-native BIO label to the unified schema."""
    if label == "O":
        return "O"
    if not (label.startswith("B-") or label.startswith("I-")):
        return "O"

    prefix, entity = label[:2], label[2:]
    mapping = _MAPS.get(source, {})
    unified = mapping.get(entity)

    if unified is None and entity not in mapping:
        key = f"{source}:{entity}"
        if key not in _UNKNOWN_WARNED:
            print(f"[label_mapping] Unknown entity '{entity}' in source '{source}' → mapped to O")
            _UNKNOWN_WARNED.add(key)
        return "O"

    if unified is None:
        return "O"

    return f"{prefix}{unified}"


def map_labels(labels: list[str], source: str) -> list[str]:
    return [map_label(l, source) for l in labels]
