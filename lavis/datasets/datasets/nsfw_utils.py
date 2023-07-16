import re
import yaml

NSFW_WORDS = [
    "cum",
    "cumming",
    "cock",
    "penis",
    "penises",
    "vaginal",
    "vagina",
    "nsfw",
    "blowjob",
    "anal",
    "nude",
    "dick",
    "pussy",
    "erection",
    "exhibitionism",
    "futanari",
    "orgasm",
    "creampie",
    "naked",
    "nipples",
    "nipple",
    "deepthroat",
    "stripper",
    "topless",
    "oral",
    "hentai",
    "erotic",
    "fellatio",
    "tits",
    "piss",
    "no clothes",
    "sex",
    "hymen",
    "exposed ass",
    "double penetration",
    "orgasm",
    "masturbation",
    "ejaculation",
    "dick",
    "semen",
    "testicle",
    "no panties",
    "ass",
    "anal",
]


def read_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def multi_split(text, delimiters):
    if text is None:
        return []
    if not delimiters:
        return [text]

    pattern = "|".join(filter(None, map(re.escape, delimiters)))
    parts = re.split(pattern, text)
    return [part.strip() for part in parts if part.strip()]


def split_words(prompt):
    return [word.strip() for word in prompt.split(",")]


def check_subsequence(array, subsequence):
    n = len(array)
    m = len(subsequence)

    for i in range(n - m + 1):
        if array[i:i + m] == subsequence:
            return True

    return False


# NSFW_WORDS = [
#     multi_split(word.lower(), [" ", "-", "_"])
#     for word in read_yaml("nsfw.yaml")
# ]
NSFW_SINGLE_WORDS = set(
    map(lambda x: x[0], filter(lambda x: len(x) == 1, NSFW_WORDS)))
NSFW_MULTI_WORDS = list(filter(lambda x: len(x) > 1, NSFW_WORDS))


def is_nsfw_word(word):
    tokens = multi_split(word, [" ", "-", "_"])
    # check single first
    has_nsfw = any(token in NSFW_SINGLE_WORDS for token in tokens)
    if has_nsfw:
        return True
    return any(
        check_subsequence(tokens, nsfw_word) for nsfw_word in NSFW_MULTI_WORDS)


def is_nsfw_prompt(prompt):
    prompt = prompt.lower()
    words = split_words(prompt)
    return any(is_nsfw_word(word) for word in words)
