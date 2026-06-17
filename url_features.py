import re
from urllib.parse import urlparse

def extract_features(url):

    features = []

    # 1 number of dots
    features.append(url.count("."))

    # 2 url length
    features.append(len(url))

    # 3 number of dash
    features.append(url.count("-"))

    # 4 number of underscore
    features.append(url.count("_"))

    # 5 number of slash
    features.append(url.count("/"))

    # 6 number of question mark
    features.append(url.count("?"))

    # 7 number of equal
    features.append(url.count("="))

    # 8 number of at symbol
    features.append(url.count("@"))

    # 9 number of digits
    features.append(sum(c.isdigit() for c in url))

    # 10 https check
    features.append(1 if "https" in url else 0)

    # Fill remaining features to reach 48
    while len(features) < 48:
        features.append(0)

    return features