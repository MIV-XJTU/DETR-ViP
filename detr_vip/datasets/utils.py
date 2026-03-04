import spacy

nlp = spacy.load("en_core_web_sm")

def extract_head_noun(phrase):
    if len(phrase) == 0:
        return None
    doc = nlp(phrase)
    for token in doc:
        # 寻找作为主语/宾语的名词（忽略修饰词）
        if token.dep_ in ("pobj", "dobj", "nsubj", "attr") and token.pos_ == "NOUN":
            return token.text
        # 如果是简单名词短语，直接返回名词
        if token.dep_ == "ROOT" and token.pos_ == "NOUN":
            return token.text
    # fallback
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return nouns[-1] if nouns else token.text

def is_noun(phrase):
    if phrase is None:
        return False
    doc = nlp(phrase)
    for token in doc:
        if token.pos_ == 'PRON':
            return False
    return True

def get_lemma(phrase):
    if phrase is None:
        return None
    doc = nlp(phrase)
    return (doc[0].lemma_).lower()