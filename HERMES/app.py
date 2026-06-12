from flask import Flask, request, jsonify, render_template
import pandas as pd
import csv
import re
import os
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
try:
    from thefuzz import fuzz, process as fuzz_process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("WARNING: thefuzz not installed. Fuzzy matching disabled. Run: pip install thefuzz")

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================
# MEDICAL SYNONYM MAP
# Keep synonyms tightly related — avoid
# generic terms that appear in many diseases
# ============================================
MEDICAL_SYNONYMS = {
    'abdominal pain':       ['stomach ache', 'stomach pain', 'belly pain', 'gastric pain', 'abdominal cramps'],
    'belly pain':           ['stomach ache', 'abdominal pain', 'gastric pain', 'stomach cramps'],
    'dizziness':            ['vertigo', 'lightheadedness', 'giddiness', 'loss of balance'],
    'fever':                ['pyrexia', 'high fever', 'feverish', 'hyperthermia'],
    'high fever':           ['pyrexia', 'hyperthermia', 'high temperature', 'fever'],
    'headache':             ['head pain', 'migraine', 'cephalalgia'],
    'difficulty breathing': ['dyspnea', 'shortness of breath', 'breathlessness', 'wheezing'],
    'chest pain':           ['thoracic pain', 'angina', 'chest discomfort', 'chest tightness'],
    'cough':                ['tussis', 'dry cough', 'wet cough', 'persistent cough'],
    'nausea':               ['queasiness', 'stomach upset'],
    'vomiting':             ['emesis', 'regurgitation'],
    'fatigue':              ['tiredness', 'exhaustion', 'weakness', 'lethargy'],
    'itching':              ['pruritus', 'skin itch', 'itchy skin'],
    'skin rash':            ['dermatitis', 'skin eruption', 'skin lesion', 'skin patches'],
    'joint pain':           ['arthralgia', 'joint ache', 'joint swelling', 'stiffness'],
    'back pain':            ['dorsalgia', 'back ache', 'lumbar pain'],
    'burning urination':    ['dysuria', 'painful urination', 'urinary pain'],
    'frequent urination':   ['polyuria', 'urinary frequency'],
    'loss of appetite':     ['anorexia', 'poor appetite'],
    'weight loss':          ['unexplained weight loss'],
    'blurred vision':       ['visual disturbance', 'blurry vision', 'vision problems'],
    'muscle pain':          ['myalgia', 'muscle ache', 'muscle soreness', 'body pain'],
    'yellowing skin':       ['jaundice', 'yellow skin', 'icterus'],
    'constipation':         ['difficult bowel', 'hard stool'],
    'diarrhea':             ['loose stool', 'watery stool', 'diarrhoea'],
    'anxiety':              ['nervousness', 'restlessness'],
    'palpitations':         ['fast heartbeat', 'irregular heartbeat'],
    'acidity':              ['acid reflux', 'heartburn', 'GERD', 'indigestion'],
    'runny nose':           ['nasal discharge', 'sneezing', 'congestion'],
    'sore throat':          ['throat pain', 'difficulty swallowing'],
    'stiff neck':           ['neck stiffness', 'neck pain'],
    'numbness':             ['tingling', 'pins and needles'],
    'skin patches':         ['skin discoloration', 'skin lesions', 'skin spots'],
    'eye pain':             ['ocular pain', 'pain behind the eyes', 'sore eyes', 'eye ache', 'painful eyes', 'eye discomfort'],
    'pain behind the eyes': ['eye pain', 'ocular pain', 'sore eyes', 'eye ache', 'periorbital pain'],
    'sore eyes':            ['eye pain', 'ocular pain', 'conjunctivitis', 'eye irritation', 'red eyes'],
    'red eyes':             ['conjunctivitis', 'eye redness', 'sore eyes', 'eye irritation'],
    'watery eyes':          ['eye discharge', 'tearing', 'lacrimation'],
}

# ============================================
# LOAD DATASETS
# ============================================
def load_filipino_dictionary(path):
    dictionary = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filipino = row['Filipino_Expression'].strip().lower()
            english = row['English_Equivalent'].strip().lower()
            dictionary[filipino] = english
    return dictionary

def load_medical_knowledge_base(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['All_Symptoms', 'Disease', 'Description'])
    df = df.reset_index(drop=True)
    return df

def normalize_symptom_text(raw):
    """
    The dataset stores symptoms as comma-separated individual words
    e.g. 'itching, skin, rash' — we join them back into a readable
    string for matching against multi-word query terms.
    We also clean up filler words like 'from', 'of', 'on'.
    """
    STOP = {'from', 'of', 'on', 'the', 'a', 'an', 'and', 'in', 'to'}
    tokens = [t.strip() for t in raw.split(',')]
    tokens = [t for t in tokens if t and t not in STOP]
    return ' '.join(tokens)

print("Loading datasets...")
FILIPINO_DICT = load_filipino_dictionary(
    os.path.join(BASE_DIR, 'data', 'filipino_symptom_dictionary_clean.csv'))
MEDICAL_DB = load_medical_knowledge_base(
    os.path.join(BASE_DIR, 'data', 'medical_knowledge_base_clean.csv'))

# ============================================
# BUILD DISEASE PROFILES
# Normalize symptom text so multi-word terms
# like "abdominal pain" are preserved
# ============================================
print("Building disease symptom profiles...")
DISEASE_PROFILES = {}
for _, row in MEDICAL_DB.iterrows():
    disease = row['Disease']
    normalized = normalize_symptom_text(row['All_Symptoms'])
    if disease not in DISEASE_PROFILES:
        DISEASE_PROFILES[disease] = {
            'symptom_text': set(),
            'description': row['Description']
        }
    DISEASE_PROFILES[disease]['symptom_text'].add(normalized)

# ============================================
# BUILD TF-IDF on NORMALIZED symptom text
# ============================================
print("Building TF-IDF index...")
MEDICAL_DB['normalized_symptoms'] = MEDICAL_DB['All_Symptoms'].apply(normalize_symptom_text)
VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
TFIDF_MATRIX = VECTORIZER.fit_transform(MEDICAL_DB['normalized_symptoms'])

print(f"HERMES ready! {len(FILIPINO_DICT)} Filipino expressions, "
      f"{len(MEDICAL_DB)} records, {len(DISEASE_PROFILES)} diseases")

# ============================================
# PATHOGNOMONIC WEIGHTING
# Count how many diseases each symptom appears in.
# Symptoms in fewer diseases = more specific = higher weight.
# ============================================
print("Computing symptom specificity weights...")
from collections import defaultdict

SYMPTOM_DISEASE_COUNT = defaultdict(set)
for _, row in MEDICAL_DB.iterrows():
    disease = row['Disease']
    tokens = [t.strip().lower() for t in row['All_Symptoms'].split(',') if t.strip()]
    for tok in tokens:
        SYMPTOM_DISEASE_COUNT[tok].add(disease)

TOTAL_DISEASES = len(DISEASE_PROFILES)

def pathognomonic_weight(term):
    """
    Returns a multiplier for a term based on how many diseases it appears in.
    Appears in 1-2 diseases  → weight 3.0  (highly specific)
    Appears in 3-5 diseases  → weight 2.0
    Appears in 6-10 diseases → weight 1.5
    Appears in 10+ diseases  → weight 1.0  (generic, no boost)
    """
    count = len(SYMPTOM_DISEASE_COUNT.get(term.lower(), set()))
    if count == 0:
        return 1.5   # unknown term, slight boost over generic
    if count <= 2:
        return 3.0
    if count <= 5:
        return 2.0
    if count <= 10:
        return 1.5
    return 1.0

# ============================================
# STEP 1: FILIPINO TO ENGLISH MAPPING
# Now with fuzzy matching for typo tolerance
# ============================================
def map_filipino_to_english(user_input):
    user_input_lower = user_input.strip().lower()

    # Exact match
    if user_input_lower in FILIPINO_DICT:
        return FILIPINO_DICT[user_input_lower], 'exact'

    # Longest partial match
    best_match, best_len = None, 0
    for filipino, english in FILIPINO_DICT.items():
        if filipino in user_input_lower and len(filipino) > best_len:
            best_match, best_len = english, len(filipino)
    if best_match:
        return best_match, 'partial'

    # Subset word match — all input words appear in a dictionary key
    # Handles dropped particles: "masakit mata" matches "masakit ang mata"
    # Prefer the SHORTEST key that contains all input words (fewest extra words = closest match)
    input_words = set(user_input_lower.split())
    best_subset_match, best_subset_len = None, float('inf')
    for filipino, english in FILIPINO_DICT.items():
        fil_words = set(filipino.split())
        if input_words and input_words.issubset(fil_words) and len(filipino) < best_subset_len:
            best_subset_match, best_subset_len = english, len(filipino)
    if best_subset_match:
        return best_subset_match, 'partial'

    # Word overlap match (require at least 2 words overlap)
    input_words = set(user_input_lower.split())
    best_score, best_english = 0, None
    for filipino, english in FILIPINO_DICT.items():
        overlap = len(input_words & set(filipino.split()))
        if overlap > best_score and overlap >= 2:
            best_score, best_english = overlap, english
    if best_english:
        return best_english, 'word_match'

    # Fuzzy match — catches typos (e.g. "tyan" → "tiyan")
    # Tiered: >=82 = confident fuzzy match (auto-applied)
    #         70-81 = low-confidence fuzzy match (applied, but flagged
    #                 to the UI as 'fuzzy_low' so the user can verify)
    if FUZZY_AVAILABLE:
        all_keys = list(FILIPINO_DICT.keys())
        best_fuzzy, score = fuzz_process.extractOne(
            user_input_lower, all_keys, scorer=fuzz.token_set_ratio
        )
        if score >= 82:
            return FILIPINO_DICT[best_fuzzy], 'fuzzy'
        if score >= 70:
            return FILIPINO_DICT[best_fuzzy], 'fuzzy_low'

    return user_input_lower, 'no_match'


# ============================================
# STEP 1b: TOKEN/BIGRAM FALLBACK
# When a full symptom phrase has no good match,
# break it into individual words and bigrams,
# filter out grammar particles/fillers, and try
# mapping each chunk against the Filipino dictionary.
# Handles long, conversational sentences like:
#   "Gabi-gabi sumasakit ang likod ko tapos parang
#    may tumutusok sa tagiliran"
# ============================================
FILIPINO_STOPWORDS = {
    # particles / markers
    'ng', 'nang', 'ang', 'mga', 'sa', 'na', 'ay', 'ko', 'mo', 'niya',
    'namin', 'natin', 'nila', 'akin', 'iyo', 'kanya', 'amin', 'atin',
    'kanila', 'po', 'si', 'kay', 'din', 'rin', 'lang', 'lamang',
    # conjunctions / connectors / fillers
    'at', 'tapos', 'pero', 'kasi', 'parang', 'tsaka', 'pati', 'kasama',
    'kapag', 'kung', 'dahil', 'para', 'tulad', 'gaya',
    # intensifiers / time words
    'sobrang', 'masyadong', 'medyo', 'paminsan', 'minsan',
    'gabi-gabi', 'araw-araw', 'tuwing', 'lagi', 'palagi',
    # english fillers
    'the', 'a', 'an', 'and', 'is', 'my', 'i', 'feel', 'feeling', 'has',
    'have', 'had', 'so', 'very', 'really', 'then', 'but', 'like',
}

def _strip_stopwords(tokens):
    return [t for t in tokens if t not in FILIPINO_STOPWORDS and len(t) > 1]


def decompose_and_map(user_input):
    """
    Token/bigram fallback for long or conversational input that doesn't
    match the dictionary as a whole phrase.

    Returns a list of (english_term, match_type) tuples for each
    recognized chunk, or an empty list if nothing was recognized.
    """
    raw_tokens = user_input.strip().lower().split()
    tokens = _strip_stopwords(raw_tokens)
    if not tokens:
        return []

    matches = []
    seen_english = set()
    used = set()

    # Try bigrams first (more specific), then unigrams for leftover tokens
    for i in range(len(tokens) - 1):
        if i in used or (i + 1) in used:
            continue
        bigram = f"{tokens[i]} {tokens[i+1]}"
        eng, mtype = map_filipino_to_english(bigram)
        if mtype in ('exact', 'partial', 'word_match', 'fuzzy'):
            if eng.lower() not in seen_english:
                matches.append((eng, mtype))
                seen_english.add(eng.lower())
            used.add(i)
            used.add(i + 1)

    for i, tok in enumerate(tokens):
        if i in used:
            continue
        eng, mtype = map_filipino_to_english(tok)
        if mtype in ('exact', 'partial', 'word_match', 'fuzzy'):
            if eng.lower() not in seen_english:
                matches.append((eng, mtype))
                seen_english.add(eng.lower())

    return matches
# Returns terms + a definition map for the UI
# ============================================

# Curated definitions for the medical synonym map keys
# Used as fallback when WordNet has no good definition
CURATED_DEFINITIONS = {
    # --- Abdominal / Digestive ---
    'abdominal pain':       ('Pain or discomfort felt in the area between the chest and pelvis.', 'Digestive / General'),
    'stomach ache':         ('A common term for pain or discomfort felt in the stomach area.', 'Digestive'),
    'stomach pain':         ('Pain or cramping felt in the stomach or surrounding area.', 'Digestive'),
    'belly pain':           ('An informal term for pain or discomfort in the stomach or abdominal area.', 'Digestive / General'),
    'gastric pain':         ('Pain originating from the stomach, often related to digestion issues.', 'Digestive'),
    'abdominal cramps':     ('Tight, cramping pain in the abdominal region, often coming in waves.', 'Digestive'),
    'stomach cramps':       ('Sudden, sharp pain or tightening in the stomach muscles.', 'Digestive'),
    'nausea':               ('An uneasy feeling in the stomach that comes with an urge to vomit.', 'Digestive'),
    'queasiness':           ('A mild, unsettled feeling in the stomach, similar to light nausea.', 'Digestive'),
    'vomiting':             ('The forceful expulsion of stomach contents through the mouth.', 'Digestive'),
    'emesis':               ('The medical term for vomiting, referring to the expulsion of stomach contents.', 'Digestive'),
    'regurgitation':        ('The bringing up of undigested food or fluid back into the mouth.', 'Digestive'),
    'constipation':         ('Difficulty passing stools, or having fewer than 3 bowel movements per week.', 'Digestive'),
    'difficult bowel':      ('Straining or discomfort when trying to have a bowel movement.', 'Digestive'),
    'hard stool':           ('Dry, firm stools that are difficult to pass, often a sign of constipation.', 'Digestive'),
    'diarrhea':             ('Frequent, loose, or watery bowel movements that may indicate infection.', 'Digestive'),
    'diarrhoea':            ('The British spelling of diarrhea. Frequent loose or watery stools.', 'Digestive'),
    'loose stool':          ('Stools that are softer or more watery than normal.', 'Digestive'),
    'watery stool':         ('Very liquid bowel movements, often a sign of digestive infection.', 'Digestive'),
    'loss of appetite':     ('A reduced or absent desire to eat, which can be caused by many illnesses.', 'Digestive / General'),
    'poor appetite':        ('Eating less than usual due to reduced hunger or interest in food.', 'Digestive / General'),
    'anorexia':             ('The medical term for loss of appetite. Not the same as the eating disorder.', 'Digestive / General'),
    'acidity':              ('Excess stomach acid that causes a burning feeling in the chest or throat.', 'Digestive'),
    'acid reflux':          ('When stomach acid flows back into the food pipe, causing a burning sensation.', 'Digestive'),
    'heartburn':            ('A burning feeling in the chest caused by stomach acid rising upward.', 'Digestive'),
    'indigestion':          ('Discomfort or pain in the upper stomach area, often after eating.', 'Digestive'),
    'GERD':                 ('Gastroesophageal Reflux Disease. A long-term condition where acid frequently rises from the stomach.', 'Digestive'),

    # --- Neurological ---
    'headache':             ('Pain or pressure felt in the head, scalp, or neck area.', 'Neurological'),
    'head pain':            ('General pain in or around the head, ranging from mild to severe.', 'Neurological'),
    'migraine':             ('A severe, recurring headache often with nausea, vomiting, or sensitivity to light.', 'Neurological'),
    'cephalalgia':          ('The medical term for headache, referring to any pain in the head region.', 'Neurological'),
    'dizziness':            ('A feeling of lightheadedness, unsteadiness, or as if the room is spinning.', 'Neurological'),
    'vertigo':              ('A spinning sensation where you or your surroundings feel like they are moving.', 'Neurological'),
    'lightheadedness':      ('A faint or woozy feeling, as if you might pass out.', 'Neurological'),
    'giddiness':            ('A casual term for lightheadedness or a feeling of being unsteady.', 'Neurological'),
    'loss of balance':      ('Difficulty maintaining a steady, upright position while standing or walking.', 'Neurological'),
    'numbness':             ('A loss of feeling or sensation in part of the body.', 'Neurological'),
    'tingling':             ('A prickling or "pins and needles" feeling, usually in the hands or feet.', 'Neurological'),
    'pins and needles':     ('A prickling sensation caused by pressure on or reduced blood flow to a nerve.', 'Neurological'),

    # --- Respiratory ---
    'difficulty breathing': ('Labored or uncomfortable breathing where getting enough air feels hard.', 'Respiratory'),
    'dyspnea':              ('The medical term for shortness of breath or difficulty breathing.', 'Respiratory'),
    'shortness of breath':  ('A feeling of not being able to breathe deeply or get enough air.', 'Respiratory'),
    'breathlessness':       ('Feeling out of breath even with little or no physical effort.', 'Respiratory'),
    'wheezing':             ('A high-pitched whistling sound when breathing, caused by narrow airways.', 'Respiratory'),
    'cough':                ('A sudden, forceful expulsion of air from the lungs to clear the airway.', 'Respiratory'),
    'tussis':               ('The medical term for cough.', 'Respiratory'),
    'dry cough':            ('A cough that produces no mucus, often caused by irritation or infection.', 'Respiratory'),
    'wet cough':            ('A cough that brings up mucus or phlegm from the airways.', 'Respiratory'),
    'persistent cough':     ('A cough that lasts more than 3 weeks and does not go away on its own.', 'Respiratory'),
    'runny nose':           ('Excess fluid draining from the nose, common in colds and allergies.', 'Respiratory'),
    'nasal discharge':      ('Fluid coming from the nose, which may be clear, yellow, or green.', 'Respiratory'),
    'sneezing':             ('A sudden, forceful expulsion of air through the nose and mouth.', 'Respiratory'),
    'congestion':           ('A blocked or stuffy feeling in the nose due to swollen nasal passages.', 'Respiratory'),
    'sore throat':          ('Pain, scratchiness, or irritation in the throat, often worse when swallowing.', 'Respiratory / ENT'),
    'throat pain':          ('Discomfort or pain felt in the throat area.', 'Respiratory / ENT'),
    'difficulty swallowing':('Trouble or pain when trying to swallow food, liquid, or saliva.', 'Respiratory / ENT'),

    # --- Cardiovascular ---
    'chest pain':           ('Discomfort or pain in the chest that may involve the heart, lungs, or muscles.', 'Cardiovascular'),
    'thoracic pain':        ('Pain in the thoracic (chest) region of the body.', 'Cardiovascular'),
    'angina':               ('Chest pain or tightness caused by reduced blood flow to the heart.', 'Cardiovascular'),
    'chest discomfort':     ('An uncomfortable pressure, squeezing, or fullness in the chest area.', 'Cardiovascular'),
    'chest tightness':      ('A feeling of pressure or squeezing in the chest, sometimes linked to the heart.', 'Cardiovascular'),
    'palpitations':         ('An awareness of your own heartbeat, which may feel fast, fluttering, or irregular.', 'Cardiovascular'),
    'fast heartbeat':       ('A heart rate that is faster than normal, sometimes felt as pounding in the chest.', 'Cardiovascular'),
    'irregular heartbeat':  ('A heartbeat that is uneven, skipping, or out of its normal rhythm.', 'Cardiovascular'),

    # --- General / Immune ---
    'fever':                ('A body temperature above 37.5 degrees Celsius, often a sign of infection.', 'General / Immune'),
    'high fever':           ('A significantly elevated body temperature, usually above 39 degrees Celsius.', 'General / Immune'),
    'feverish':             ('Feeling like you have a fever, with warmth, chills, or general discomfort.', 'General / Immune'),
    'pyrexia':              ('The medical term for fever, meaning an abnormal rise in body temperature.', 'General / Immune'),
    'hyperthermia':         ('A dangerously high body temperature, often caused by heat exposure.', 'General / Immune'),
    'high temperature':     ('Body temperature that is higher than the normal range of 36 to 37 degrees Celsius.', 'General / Immune'),
    'fatigue':              ('Persistent tiredness or low energy that does not go away with rest.', 'General'),
    'tiredness':            ('A general feeling of being worn out or lacking energy.', 'General'),
    'exhaustion':           ('Extreme tiredness that makes it hard to carry out everyday tasks.', 'General'),
    'weakness':             ('A lack of physical strength or energy, often felt throughout the body.', 'General'),
    'lethargy':             ('Unusual drowsiness or a general lack of energy and motivation.', 'General / Neurological'),
    'weight loss':          ('An unintentional drop in body weight that may be a sign of an underlying illness.', 'General / Endocrine'),
    'unexplained weight loss': ('Losing weight without trying or without a clear reason, which may need medical attention.', 'General / Endocrine'),
    'anxiety':              ('Excessive worry, fear, or nervousness that feels hard to control.', 'Psychological'),
    'nervousness':          ('A feeling of being uneasy, worried, or on edge.', 'Psychological'),
    'restlessness':         ('Difficulty staying still or feeling calm, often linked to anxiety or discomfort.', 'Psychological'),

    # --- Musculoskeletal ---
    'joint pain':           ('Aching, soreness, or discomfort in any joint of the body.', 'Musculoskeletal'),
    'arthralgia':           ('The medical term for joint pain, without necessarily involving inflammation.', 'Musculoskeletal'),
    'joint ache':           ('A dull, persistent discomfort in one or more joints.', 'Musculoskeletal'),
    'joint swelling':       ('Swelling around a joint, often caused by fluid buildup or inflammation.', 'Musculoskeletal'),
    'stiffness':            ('Tightness or reduced movement in a joint or muscle, especially after rest.', 'Musculoskeletal'),
    'back pain':            ('Pain felt in the lower, middle, or upper back.', 'Musculoskeletal'),
    'back ache':            ('A dull or aching pain in the back, often in the lower region.', 'Musculoskeletal'),
    'dorsalgia':            ('The medical term for back pain, referring to pain along the spine.', 'Musculoskeletal'),
    'lumbar pain':          ('Pain in the lower back, in the lumbar region of the spine.', 'Musculoskeletal'),
    'muscle pain':          ('Aching or soreness in one or more muscles, common after illness or overuse.', 'Musculoskeletal'),
    'myalgia':              ('The medical term for muscle pain or tenderness.', 'Musculoskeletal'),
    'muscle ache':          ('A dull aching sensation in the muscles, often felt during illness or after activity.', 'Musculoskeletal'),
    'muscle soreness':      ('Tenderness or discomfort in the muscles, usually after physical exertion.', 'Musculoskeletal'),
    'body pain':            ('General aching or discomfort felt across multiple areas of the body.', 'Musculoskeletal'),
    'stiff neck':           ('Tightness or reduced movement in the neck, sometimes a sign of serious conditions like meningitis.', 'Musculoskeletal / Neurological'),
    'neck stiffness':       ('Difficulty moving the neck freely, often with pain or tightness.', 'Musculoskeletal / Neurological'),
    'neck pain':            ('Pain or discomfort felt in the neck area.', 'Musculoskeletal'),

    # --- Dermatological ---
    'itching':              ('An irritating sensation on the skin that creates an urge to scratch.', 'Dermatological'),
    'pruritus':             ('The medical term for itching, referring to skin irritation that triggers scratching.', 'Dermatological'),
    'skin itch':            ('An itch felt on the surface of the skin.', 'Dermatological'),
    'itchy skin':           ('Skin that feels persistently irritated, causing the urge to scratch.', 'Dermatological'),
    'skin rash':            ('A visible change in skin color, texture, or appearance, often with redness or irritation.', 'Dermatological'),
    'dermatitis':           ('Inflammation of the skin that causes redness, itching, and sometimes blisters.', 'Dermatological'),
    'skin eruption':        ('A sudden appearance of spots, bumps, or redness on the skin.', 'Dermatological'),
    'skin lesion':          ('Any abnormal area on the skin, such as a sore, blister, or patch.', 'Dermatological'),
    'skin patches':         ('Areas of skin with a different color or texture from the surrounding skin.', 'Dermatological'),
    'skin discoloration':   ('A change in the normal color of the skin, which may be lighter or darker.', 'Dermatological'),
    'skin lesions':         ('Abnormal areas on the skin surface, including sores, bumps, or discolored patches.', 'Dermatological'),
    'skin spots':           ('Small, distinct marks or discolored areas on the skin surface.', 'Dermatological'),
    'yellowing skin':       ('A yellow tint to the skin and eyes, usually caused by liver or bile problems.', 'Hepatic'),
    'jaundice':             ('A yellowing of the skin and whites of the eyes due to high bilirubin levels.', 'Hepatic'),
    'yellow skin':          ('Skin that appears yellow, often a sign of jaundice or liver-related conditions.', 'Hepatic'),
    'icterus':              ('The medical term for jaundice, describing yellow discoloration from bile pigment buildup.', 'Hepatic'),

    # --- Urological ---
    'burning urination':    ('A burning or stinging feeling during urination, often caused by infection.', 'Urological'),
    'dysuria':              ('The medical term for painful or difficult urination.', 'Urological'),
    'painful urination':    ('Discomfort or pain felt when passing urine, commonly linked to UTI.', 'Urological'),
    'urinary pain':         ('Any pain or discomfort experienced in the urinary tract during urination.', 'Urological'),
    'frequent urination':   ('Urinating more often than usual, which may indicate infection or other conditions.', 'Urological'),
    'polyuria':             ('The production of an unusually large amount of urine, sometimes linked to diabetes.', 'Urological / Endocrine'),
    'urinary frequency':    ('Needing to urinate more often than normal throughout the day.', 'Urological'),

    # --- Ophthalmological ---
    'blurred vision':       ('Objects appearing out of focus or unclear, making it hard to see properly.', 'Ophthalmological'),
    'visual disturbance':   ('Any change in normal vision, such as blurring, spots, or loss of sight.', 'Ophthalmological'),
    'blurry vision':        ('A reduction in the sharpness of vision where things look fuzzy or unclear.', 'Ophthalmological'),
    'vision problems':      ('Any difficulty with sight, including blurring, double vision, or loss of vision.', 'Ophthalmological'),
}

def get_definition(term):
    """
    Universal definition lookup for any term.
    Priority: (1) CURATED_DEFINITIONS exact match
              (2) WordNet — any synset whose definition contains a
                  medical keyword, using any part of speech
              (3) WordNet — first available synset regardless of topic
    Returns (definition_string, body_system_string) or None.
    """
    # 1. Curated dictionary — fastest, most accurate
    if term in CURATED_DEFINITIONS:
        return CURATED_DEFINITIONS[term]

    MEDICAL_KEYWORDS = {
        'pain', 'disease', 'disorder', 'symptom', 'condition', 'infection',
        'inflammation', 'fever', 'swelling', 'body', 'medical', 'clinical',
        'illness', 'injury', 'nerve', 'muscle', 'skin', 'blood', 'heart',
        'lung', 'stomach', 'liver', 'kidney', 'bone', 'joint', 'throat',
        'temperature', 'breathing', 'digestion', 'vision', 'sensation',
        'nausea', 'vomit', 'cough', 'rash', 'itch', 'fatigue', 'sweat',
        'chill', 'spasm', 'cramp', 'bleed', 'bruise', 'swell', 'toxic',
    }

    # 2. Try all parts of speech — medical hit preferred
    all_synsets = (
        wordnet.synsets(term) +
        wordnet.synsets(term.replace(' ', '_'))
    )
    medical_hit = None
    any_hit = None
    for syn in all_synsets:
        defn = syn.definition()
        if any_hit is None:
            any_hit = defn
        if medical_hit is None and any(kw in defn.lower() for kw in MEDICAL_KEYWORDS):
            medical_hit = defn
            break

    chosen = medical_hit or any_hit
    if chosen:
        return (chosen.capitalize().rstrip('.') + '.', 'General')

    # 3. Try individual words of a multi-word term
    words = [w for w in term.split() if len(w) > 3]
    for word in words:
        for syn in wordnet.synsets(word)[:3]:
            defn = syn.definition()
            if any(kw in defn.lower() for kw in MEDICAL_KEYWORDS):
                return (defn.capitalize().rstrip('.') + '.', 'General')

    return None


def expand_query(english_term):
    """
    Returns:
        terms      : list of expanded term strings, ranked by specificity
        definitions: dict mapping term -> (definition, body_system)
        sources    : dict mapping term -> source label string
    """
    expanded = {}   # term -> source label
    definitions = {}

    def add_with_def(term, source):
        if term not in expanded:
            expanded[term] = source
        defn = get_definition(term)
        if defn:
            definitions[term] = defn

    # Root term — from dictionary mapping
    expanded[english_term] = 'Dictionary Match'
    root_def = get_definition(english_term)
    if root_def:
        definitions[english_term] = root_def

    # Curated medical synonyms
    if english_term in MEDICAL_SYNONYMS:
        for syn in MEDICAL_SYNONYMS[english_term]:
            add_with_def(syn, 'Medical Synonym')

    # Partial key match in synonym map
    for key, synonyms in MEDICAL_SYNONYMS.items():
        if key != english_term and (key in english_term or english_term in key):
            add_with_def(key, 'Medical Synonym')
            for syn in synonyms[:2]:
                add_with_def(syn, 'Medical Synonym')

    # WordNet expansion
    # Only expand words that are medical/symptom roots — never positional,
    # directional, or preposition words that produce anatomical false positives
    # (e.g. "behind" → "buttocks", "over" → body parts, etc.)
    WORDNET_SKIP_WORDS = {
        # Prepositions / directions that mislead WordNet into anatomy
        'behind', 'below', 'above', 'under', 'over', 'around', 'beside',
        'front', 'back', 'side', 'left', 'right', 'upper', 'lower',
        # Generic symptom modifiers — too broad to expand meaningfully
        'pain', 'ache', 'hurt', 'sore', 'feeling', 'sensation',
    }
    NOISE_TERMS = {
        'condition', 'state', 'result', 'process', 'system',
        'structure', 'form', 'type', 'kind', 'manner',
        # Anatomical false positives from positional words
        'buttocks', 'nates', 'rear', 'bottom', 'rump', 'seat',
        'hurting', 'painfulness',
    }
    MEDICAL_KEYWORDS = {
        'pain', 'disease', 'disorder', 'symptom', 'condition',
        'infection', 'inflammation', 'fever', 'swelling', 'body',
        'organ', 'tissue', 'nerve', 'muscle', 'gland', 'vessel',
    }
    # Only expand content words (body part names, symptom nouns)
    # that are likely to have useful medical synonyms in WordNet
    for word in english_term.split():
        if len(word) <= 3:
            continue
        if word.lower() in WORDNET_SKIP_WORDS:
            continue
        for syn in wordnet.synsets(word, pos=wordnet.NOUN)[:2]:
            defn = syn.definition().lower()
            if any(kw in defn for kw in MEDICAL_KEYWORDS):
                for lemma in syn.lemmas()[:2]:
                    term = lemma.name().replace('_', ' ').lower()
                    if len(term) > 3 and term not in NOISE_TERMS and term != word:
                        add_with_def(term, 'WordNet')

    # ── Rank terms by pathognomonic weight (most specific first) ──
    ranked = sorted(
        expanded.keys(),
        key=lambda t: pathognomonic_weight(t),
        reverse=True
    )

    # Build sources dict for UI
    sources = {term: expanded[term] for term in ranked}

    return ranked, definitions, sources


# ============================================
# CONFIDENCE TIER HELPER
# ============================================
def confidence_tier(pct):
    """Returns tier label based on confidence percentage."""
    if pct >= 60:
        return 'high'
    elif pct >= 30:
        return 'medium'
    else:
        return 'low'

# ============================================
# DISEASE CATEGORY RULES  (Fix 4)
# Infectious diseases require at least one
# "infectious marker" symptom in the user's
# input before they can appear in results.
# This prevents musculoskeletal/neurological
# inputs from pulling viral/bacterial diseases.
# ============================================
INFECTIOUS_DISEASES = {
    'Dengue', 'Malaria', 'Typhoid', 'Hepatitis A', 'Hepatitis B',
    'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Chicken pox',
    'Common Cold', 'Pneumonia', 'AIDS', 'Tuberculosis',
    'Urinary tract infection', 'Impetigo',
}

INFECTIOUS_MARKERS = {
    'fever', 'high fever', 'pyrexia', 'hyperthermia', 'feverish',
    'chills', 'rash', 'skin rash', 'vomiting', 'emesis',
    'nausea', 'diarrhea', 'diarrhoea', 'loose stool', 'watery stool',
    'loss of appetite', 'poor appetite',
    'sweating', 'night sweats', 'rigors',
}

def has_infectious_marker(all_terms):
    """Returns True if any term in all_terms is an infectious marker."""
    lower_terms = {t.lower() for t in all_terms}
    return bool(lower_terms & INFECTIOUS_MARKERS)

# ============================================
# STEP 3: RETRIEVAL  (Fixes 1, 2, 3, 4)
#
# Fix 1 — Root term must exist in disease profile to qualify
# Fix 2 — Root term gets 5x boost over expansion terms
# Fix 3 — Only direct synonyms used (1-degree expansion only)
# Fix 4 — Infectious diseases gated behind infectious markers
# ============================================
def retrieve_health_info(english_term, expanded_terms, top_n=3):
    # ── Fix 3: Only use direct synonyms (1-degree) ──────────────────
    # The expanded_terms list is already ranked — terms closest to the
    # root come first (direct synonyms from MEDICAL_SYNONYMS), then
    # WordNet expansions. We cap at direct synonyms only by only taking
    # terms that are either the root or appear as a direct value in
    # MEDICAL_SYNONYMS[english_term]. Everything else is dropped.
    direct_synonyms = set(MEDICAL_SYNONYMS.get(english_term, []))
    # Also include terms that have the root as THEIR synonym (reverse map)
    for key, syns in MEDICAL_SYNONYMS.items():
        if english_term in syns:
            direct_synonyms.add(key)

    filtered_expansions = [
        t for t in expanded_terms
        if t == english_term or t.lower() in {s.lower() for s in direct_synonyms}
    ]
    # Fall back to full list if filtering leaves nothing
    if not filtered_expansions:
        filtered_expansions = expanded_terms

    all_terms = [english_term] + [t for t in filtered_expansions if t != english_term]

    # ── Fix 2: Root term gets 5x boost ──────────────────────────────
    # Build weighted query: root repeated 5x, expansion terms by their
    # pathognomonic weight (1-3x). This anchors the result to the actual
    # symptom the user typed, not the expansion cloud.
    weighted_parts = [english_term] * 5   # 5x root boost
    for term in all_terms[1:]:            # expansion terms
        w = pathognomonic_weight(term)
        repeat = max(1, int(round(w)))
        weighted_parts.extend([term] * repeat)

    query_str = ' '.join(weighted_parts)
    qvec = VECTORIZER.transform([query_str])
    cos_scores = cosine_similarity(qvec, TFIDF_MATRIX).flatten()

    # ── Fix 1: Root term must exist in disease profile ───────────────
    # Build a set of diseases that actually contain the root term
    # (or any of its direct synonyms) in their symptom profile.
    # Diseases without any root-term presence are disqualified.
    root_terms_to_check = {english_term.lower()} | {s.lower() for s in direct_synonyms}
    qualified_diseases = set()
    for disease, profile in DISEASE_PROFILES.items():
        combined_text = ' '.join(profile['symptom_text']).lower()
        # Check if any root or direct synonym word appears in the profile
        for rt in root_terms_to_check:
            # Match individual words of the term (handles "back pain" stored as "back pain" or "back, pain")
            rt_words = rt.split()
            if all(word in combined_text for word in rt_words):
                qualified_diseases.add(disease)
                break

    # ── Fix 4: Gate infectious diseases ─────────────────────────────
    infectious_allowed = has_infectious_marker(all_terms)

    # Secondary: exact term bonus per disease, weighted
    disease_bonus = {}
    for disease, profile in DISEASE_PROFILES.items():
        combined_text = ' '.join(profile['symptom_text']).lower()
        bonus = 0
        for term in all_terms[:8]:
            if term.lower() in combined_text:
                w = pathognomonic_weight(term)
                bonus += w * 2 if f' {term.lower()} ' in f' {combined_text} ' else w
        disease_bonus[disease] = bonus

    # Combined score: cosine + normalized bonus
    max_bonus = max(disease_bonus.values()) if disease_bonus else 1
    combined = {}
    for i, row in MEDICAL_DB.iterrows():
        disease = row['Disease']

        # Fix 1: Skip disqualified diseases
        if disease not in qualified_diseases:
            continue

        # Fix 4: Skip infectious diseases if no infectious marker present
        if disease in INFECTIOUS_DISEASES and not infectious_allowed:
            continue

        bonus = disease_bonus.get(disease, 0)
        norm_bonus = (bonus / max_bonus) * 0.3 if max_bonus > 0 else 0
        combined[i] = float(cos_scores[i]) * 0.7 + norm_bonus

    if not combined:
        # Fallback: relax Fix 1 (drop root-term requirement) but keep Fix 4
        # so we still return something useful rather than empty results
        for i, row in MEDICAL_DB.iterrows():
            disease = row['Disease']
            if disease in INFECTIOUS_DISEASES and not infectious_allowed:
                continue
            bonus = disease_bonus.get(disease, 0)
            norm_bonus = (bonus / max_bonus) * 0.3 if max_bonus > 0 else 0
            combined[i] = float(cos_scores[i]) * 0.7 + norm_bonus

    if not combined:
        return []

    sorted_indices = sorted(combined, key=combined.get, reverse=True)

    seen_diseases = {}
    for idx in sorted_indices:
        disease = MEDICAL_DB.iloc[idx]['Disease']
        score = combined[idx]
        if disease not in seen_diseases:
            seen_diseases[disease] = (idx, score)

    top_diseases = sorted(seen_diseases.values(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for idx, combo_score in top_diseases:
        if combo_score < 0.05:
            continue
        row = MEDICAL_DB.iloc[idx]
        raw_cos = float(cos_scores[idx])
        pct = min(round((raw_cos / 0.65) * 90, 1), 95.0)
        pct = max(pct, 10.0)

        # Compute how many root/direct-synonym terms matched this disease
        profile = DISEASE_PROFILES.get(row['Disease'], {})
        combined_text = ' '.join(profile.get('symptom_text', [])).lower()
        matched_root_count = sum(
            1 for rt in root_terms_to_check
            if all(word in combined_text for word in rt.split())
        )
        total_query_terms = len(all_terms)

        results.append({
            'disease':            row['Disease'],
            'description':        row['Description'],
            'matched_symptoms':   row['All_Symptoms'],
            'confidence':         pct,
            # Extra data for frontend accuracy indicators
            'matched_root_count': matched_root_count,
            'total_query_terms':  total_query_terms,
            'root_term':          english_term,
            'direct_synonyms':    list(direct_synonyms),
        })

    return results


# ============================================
# SYMPTOM DELTA — FOLLOW-UP QUESTIONS
# Mimics a doctor asking clarifying questions:
# looks at the top candidate diseases and finds
# symptoms common in those diseases that the user
# has NOT mentioned (and not already denied), to
# help discriminate between close candidates.
# ============================================

def _readable_symptom(term):
    """Turn a raw symptom token/phrase into a readable label, e.g. 'high fever' -> 'High fever'."""
    return term.strip().capitalize()


# Build a vocabulary of clean, askable symptom terms once at startup.
# Sources: MEDICAL_SYNONYMS keys (curated, multi-word, human-readable)
# plus multi-word entries from SYMPTOM_DISEASE_COUNT (derived from the
# medical knowledge base's raw symptom tokens, rejoined where they form
# known phrases). Single stray tokens (e.g. "lack", "look", "(typhos)")
# are excluded — they make poor standalone questions.
_FOLLOWUP_VOCAB = set()
for _term in MEDICAL_SYNONYMS.keys():
    if len(_term.split()) >= 2:
        _FOLLOWUP_VOCAB.add(_term)
for _term in CURATED_DEFINITIONS.keys():
    if len(_term.split()) >= 2:
        _FOLLOWUP_VOCAB.add(_term)


def get_followup_questions(results, mentioned_terms, negated_terms, max_questions=5):
    """
    Given the top disease results, the terms already covered by the user's
    query (mentioned_terms), and terms they've already denied (negated_terms),
    return a ranked list of follow-up symptom questions to help discriminate
    between the top candidate diseases.

    Candidate questions are drawn from a curated vocabulary of clean,
    human-readable multi-word symptom terms (MEDICAL_SYNONYMS /
    CURATED_DEFINITIONS keys), checked against each disease's symptom text.

    Returns a list of dicts: {'term': <symptom>, 'label': <readable text>}
    """
    if not results:
        return []

    mentioned_lower = {t.lower() for t in mentioned_terms}
    negated_lower = {t.lower() for t in negated_terms}

    candidate_diseases = [r['disease'] for r in results[:3]]

    # delta_counts: symptom -> number of top diseases whose symptom text contains it
    delta_counts = defaultdict(int)
    for disease in candidate_diseases:
        profile = DISEASE_PROFILES.get(disease)
        if not profile:
            continue
        combined_text = ' '.join(profile['symptom_text']).lower()

        for term in _FOLLOWUP_VOCAB:
            if term in mentioned_lower or term in negated_lower:
                continue
            # check all words of the term appear in the disease's symptom text
            # (handles cases where the raw data stores "high, fever" but the
            # term is "high fever")
            if all(word in combined_text for word in term.split()):
                delta_counts[term] += 1

    if not delta_counts:
        return []

    # Rank: prefer symptoms appearing in FEWER of the top diseases first
    # (most discriminating), then by pathognomonic specificity.
    ranked = sorted(
        delta_counts.keys(),
        key=lambda s: (delta_counts[s], -pathognomonic_weight(s))
    )

    questions = []
    for term in ranked:
        questions.append({'term': term, 'label': _readable_symptom(term)})
        if len(questions) >= max_questions:
            break

    return questions


# RED FLAG TRIAGE
# Symptoms that may indicate a medical emergency
# ============================================
RED_FLAG_SYMPTOMS = {
    'chest pain', 'chest tightness', 'chest discomfort', 'angina',
    'difficulty breathing', 'shortness of breath', 'breathlessness', 'dyspnea',
    'loss of balance', 'numbness', 'sudden numbness',
    'blurred vision', 'sudden vision loss',
    'severe headache', 'sudden headache',
    'palpitations', 'irregular heartbeat',
    'loss of consciousness', 'fainting',
    'coughing blood', 'vomiting blood',
    'high fever',
}

def check_red_flags(english_terms):
    """
    Returns list of red flag terms found in the mapped/expanded terms.
    """
    hits = []
    for term in english_terms:
        if term.lower() in RED_FLAG_SYMPTOMS:
            hits.append(term)
    return hits


# ============================================
# NEGATION DETECTION
# Strips negated symptoms before retrieval
# ============================================
NEGATION_WORDS_FIL = {'walang', 'wala', 'hindi', 'hindi naman', 'wala naman'}
NEGATION_WORDS_EN  = {'no', 'not', 'without', 'no sign of', 'no signs of'}

def detect_negations(user_input):
    """
    Returns a set of symptom-like tokens that appear after a negation word.
    These will be excluded from the TF-IDF query.
    """
    negated = set()
    tokens = user_input.lower().split()
    all_neg = NEGATION_WORDS_FIL | NEGATION_WORDS_EN

    i = 0
    while i < len(tokens):
        # Check single token negation
        if tokens[i] in all_neg:
            # Grab up to next 3 tokens as the negated phrase
            for length in [3, 2, 1]:
                if i + length <= len(tokens):
                    phrase = ' '.join(tokens[i+1:i+1+length])
                    negated.add(phrase)
            i += 2
            continue
        # Check two-token negation phrases (e.g. "hindi naman")
        if i + 1 < len(tokens):
            two = tokens[i] + ' ' + tokens[i+1]
            if two in all_neg:
                for length in [3, 2, 1]:
                    if i + 2 + length <= len(tokens):
                        phrase = ' '.join(tokens[i+2:i+2+length])
                        negated.add(phrase)
                i += 3
                continue
        i += 1
    return negated


# ============================================
# MULTI-SYMPTOM EXTRACTION
# Splits compound input into individual symptoms
# ============================================

# Filipino/Taglish conjunctions and separators
SPLIT_PATTERNS = [
    r'\bat\b',           # "at" (Filipino "and")
    r'\btapos\b',        # "tapos" (then/and)
    r'\tpero\b',         # "pero" (but)
    r'\bpati\b',         # "pati" (also)
    r'\btsaka\b',        # "tsaka" (and also)
    r'\bkasama\b',       # "kasama" (together with)
    r'\band\b',
    r'\bplus\b',
    r',',
    r';',
]

SPLIT_REGEX = re.compile('|'.join(SPLIT_PATTERNS), re.IGNORECASE)

def extract_symptoms(user_input):
    """
    Splits user input into individual symptom candidates.
    Returns list of non-empty stripped strings.
    """
    parts = SPLIT_REGEX.split(user_input)
    symptoms = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in symptoms:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique.append(s)
    return unique if unique else [user_input.strip()]


# ============================================
# FLASK ROUTES
# ============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/expressions')
def get_expressions():
    """Returns the list of known Filipino expressions for frontend autocomplete."""
    return jsonify(sorted(FILIPINO_DICT.keys()))

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    user_input = data.get('query', '').strip()
    if not user_input:
        return jsonify({'error': 'Please enter a symptom'}), 400

    # ── Negation detection ──────────────────────────────
    negated_terms = detect_negations(user_input)

    # ── Multi-symptom extraction ────────────────────────
    symptom_parts = extract_symptoms(user_input)

    # Map each symptom part independently
    mapped_symptoms  = []
    all_expanded     = []
    all_definitions  = {}
    all_sources      = {}   # term -> source label

    for part in symptom_parts:
        eng, mtype = map_filipino_to_english(part)

        if mtype == 'no_match':
            # Fallback: try decomposing into tokens/bigrams to salvage
            # recognizable terms from a long/conversational phrase.
            decomposed = decompose_and_map(part)
            if decomposed:
                for d_eng, d_mtype in decomposed:
                    mapped_symptoms.append({'input': part, 'english': d_eng, 'match_type': 'decomposed_' + d_mtype})
                    exp_terms, defs, sources = expand_query(d_eng)
                    exp_terms = [t for t in exp_terms
                                 if t.lower() not in negated_terms
                                 and d_eng.lower() not in negated_terms]
                    all_expanded.extend(exp_terms)
                    all_definitions.update(defs)
                    all_sources.update(sources)
                continue  # skip the default no_match handling below

        mapped_symptoms.append({'input': part, 'english': eng, 'match_type': mtype})
        exp_terms, defs, sources = expand_query(eng)
        # Filter out negated terms
        exp_terms = [t for t in exp_terms
                     if t.lower() not in negated_terms
                     and eng.lower() not in negated_terms]
        all_expanded.extend(exp_terms)
        all_definitions.update(defs)
        all_sources.update(sources)

    # Deduplicate expanded terms (preserve ranked order)
    seen = set()
    unique_expanded = []
    for t in all_expanded:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_expanded.append(t)

    # Primary english term = first mapped symptom
    primary_english = mapped_symptoms[0]['english'] if mapped_symptoms else user_input.lower()
    primary_match_type = mapped_symptoms[0]['match_type'] if mapped_symptoms else 'no_match'

    # ── Red flag check ──────────────────────────────────
    red_flags = check_red_flags([m['english'] for m in mapped_symptoms] + unique_expanded)

    # Build readable multi-symptom label for the UI
    if len(mapped_symptoms) > 1:
        english_mapped_display = ' + '.join(m['english'] for m in mapped_symptoms)
    else:
        english_mapped_display = primary_english

    # ── Did You Mean? — check if any results exist at all for fallback ──
    # Run a quick retrieval to detect total no-match (nothing in DB for these terms)
    # so we can show "Did You Mean?" immediately even before follow-ups.
    quick_results = retrieve_health_info(primary_english, unique_expanded, top_n=1)
    if not quick_results:
        suggestions = []
        if FUZZY_AVAILABLE:
            all_known = list(MEDICAL_SYNONYMS.keys())
            fuzzy_hits = fuzz_process.extract(
                primary_english, all_known, scorer=fuzz.token_sort_ratio, limit=5
            )
            suggestions = [h[0] for h in fuzzy_hits if h[1] >= 50]

        return jsonify({
            'original_input':   user_input,
            'english_mapped':   english_mapped_display,
            'match_type':       primary_match_type,
            'mapped_symptoms':  mapped_symptoms,
            'expanded_terms':   unique_expanded[:12],
            'term_definitions': all_definitions,
            'term_sources':     all_sources,
            'negated_terms':    list(negated_terms),
            'red_flags':        red_flags,
            'suggestions':      suggestions,
            'found': False,
            'has_followup': False,
            'message': 'No relevant health information found. Please consult a medical professional.'
        })

    # ── Generate follow-up questions using a preliminary top-3 ──────────
    # We run retrieval here ONLY to generate relevant follow-up questions,
    # but we do NOT return the match to the frontend yet.
    prelim_results = retrieve_health_info(primary_english, unique_expanded, top_n=3)
    mentioned_terms = [m['english'] for m in mapped_symptoms] + unique_expanded
    followup_questions = get_followup_questions(prelim_results, mentioned_terms, negated_terms)

    # ── Return mapping + expansion + follow-up questions ONLY ──────────
    # The actual match result is withheld until /refine is called.
    return jsonify({
        'original_input':   user_input,
        'english_mapped':   english_mapped_display,
        'match_type':       primary_match_type,
        'mapped_symptoms':  mapped_symptoms,
        'expanded_terms':   unique_expanded[:12],
        'term_definitions': all_definitions,
        'term_sources':     all_sources,
        'negated_terms':    list(negated_terms),
        'red_flags':        red_flags,
        'followup_questions': followup_questions,
        'has_followup': True,
        # Context the frontend echoes back to /refine
        'refine_context': {
            'primary_english': primary_english,
            'expanded_terms':  unique_expanded,
            'negated_terms':   list(negated_terms),
        },
    })


@app.route('/refine', methods=['POST'])
def refine():
    """
    Recalculates results after the user answers follow-up
    (Symptom Delta) questions.

    Expects JSON:
      {
        "refine_context": { "primary_english": ..., "expanded_terms": [...], "negated_terms": [...] },
        "answers": { "<symptom_term>": "yes" | "no", ... }
      }
    """
    data = request.get_json()
    context = data.get('refine_context', {})
    answers = data.get('answers', {})

    primary_english = context.get('primary_english', '')
    expanded_terms = list(context.get('expanded_terms', []))
    negated_terms = set(context.get('negated_terms', []))

    if not primary_english:
        return jsonify({'error': 'Missing refine context'}), 400

    confirmed_terms = []
    for term, answer in answers.items():
        term_lower = term.lower().strip()
        if answer == 'yes':
            confirmed_terms.append(term_lower)
            negated_terms.discard(term_lower)
        elif answer == 'no':
            negated_terms.add(term_lower)

    # Build the enriched term list: original expanded terms + newly
    # confirmed symptoms, minus anything the user has now denied.
    combined_terms = expanded_terms + confirmed_terms
    combined_terms = [t for t in combined_terms if t.lower() not in negated_terms]

    # Deduplicate, preserve order
    seen = set()
    unique_expanded = []
    for t in combined_terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_expanded.append(t)

    if primary_english.lower() in negated_terms:
        primary_english_for_query = unique_expanded[0] if unique_expanded else primary_english
    else:
        primary_english_for_query = primary_english

    results = retrieve_health_info(primary_english_for_query, unique_expanded, top_n=3)

    red_flags = check_red_flags([primary_english_for_query] + unique_expanded)

    if not results:
        return jsonify({
            'found': False,
            'message': 'No relevant health information found. Please consult a medical professional.',
            'red_flags': red_flags,
            'confirmed_terms': confirmed_terms,
            'negated_terms': list(negated_terms),
        })

    best = results[0]
    tier = confidence_tier(best['confidence'])

    # No further follow-up round in v1 — single round only
    return jsonify({
        'found': True,
        'disease':            best['disease'],
        'description':        best['description'],
        'matched_symptoms':   best['matched_symptoms'],
        'confidence':         best['confidence'],
        'confidence_tier':    tier,
        'other_matches':      results[1:],
        'red_flags':          red_flags,
        'confirmed_terms':    confirmed_terms,
        'negated_terms':      list(negated_terms),
        'expanded_terms':     unique_expanded[:12],
        'matched_root_count': best.get('matched_root_count', 0),
        'total_query_terms':  best.get('total_query_terms', len(unique_expanded)),
        'root_term':          best.get('root_term', primary_english),
        'direct_synonyms':    best.get('direct_synonyms', []),
        'is_single_symptom':  len(confirmed_terms) == 0 and len(unique_expanded) <= 4,
    })


if __name__ == '__main__':
    app.run(debug=True)
