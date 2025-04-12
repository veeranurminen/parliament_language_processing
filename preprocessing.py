"""
Tässä tiedostossa on eduskunnan puheenvuorojen esikäsittelyyn tarvittavat
funktiot. Toimintoja:
    - Erikoismerkkien poisto
    - Lyhenteiden avaaminen ("ed." -> "edustaja")
    - Vuoden ja puhujan roolin irroittaminen omiin sarakkeisiinsa
    - Kielensuodatus
    - Lemmatisointi

Tekijä: Veera Nurminen
2025 Tampere
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import langdetect
from functools import lru_cache
import stanza
import logging

# Vähennetään Stanzan lokitietojen määrää
stanza.logger.setLevel(logging.WARNING)

# Ladataan Stanza-malli
if not os.path.exists(os.path.expanduser('~/.stanza_resources/fi')):
    stanza.download('fi')
nlp = stanza.Pipeline('fi', processors='tokenize,lemma, mwt',
                      download_method=None)


def replace_abbreviations(text):
    """
    Korvaa lyhenteen ed. -> edustaja Regex:n avulla
    :param text: str
    :return: str
    """
    # Varmistetaan, ettei käsitellä mitään mikä ei ole merkkijono
    if not isinstance(text, str):
        return text

    # "ed." -> "edustaja"
    text = re.sub(r'\bed\.\s?', 'edustaja ', text)
    # "Ed." -> "Edustaja"
    text = re.sub(r'\bEd\.\s?', 'Edustaja ', text)

    return text


def date_and_speech_preprocessing(data):
    """
    Funktio muuttaa päivämäärät pelkiksi vuosiksi, ja merkkaa onko puhuja
    puhemies (2), varapuhemies (1) vai joku muu (0)
    :param data: Numpy-array, sisältää tiedot 'date' ja 'speech' -sarakkeista
    :return: Numpy-array, päivitetyt data-sarakkeet
    """
    data[:, 0] = [int(x.split(".")[-1]) for x in data[:, 0]]

    data[:, 1] = [1 if "varapuhemies" in x.lower()
                  else (2 if "puhemies" in x.lower() else 0) for x in
                  data[:, 1]]
    return data


def clean_text(text):
    """
    Funktio poistaa erikoismerkit ja ylimääräiset välilyönnit sekä muuttaa
    tekstin pieniksi kirjaimiksi.
    :param text: str, käsiteltävä teksti
    :return: str, siivottu teksti
    """
    text = re.sub(r'[^a-zA-ZåäöÅÄÖ0-9]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.lower().strip()


def language_filtering_progress(df, column):
    """
    Funktio suodattaa vieraskieliseksi tunnistetut puheenvuorot pois ja lisää
    prosessiin edistymispalkin.
    :param df: DataFrame, tiedoston tiedot
    :param column: str, sarakkeen nimi
    :return: DataFrame, päivitetty DataFrame
    """
    # Edistymispalkki
    tqdm.pandas(desc="Kielensuodatus käynnissä")
    # Kielensuodatus
    df[column] = df[column].progress_apply(filter_finnish_words_cached)

    # Poistetaan tiedoston rivit, joissa suodatuksen tulos on None
    df = df.dropna(subset=[column]).reset_index(drop=True)

    return df


def filter_finnish_words_cached(text):
    """
    Funktion avulla tiedostosta poistetaan ne puheenvuorot, joissa tunnistetaan
    olevan raja-arvoa suurempi osuus muuta kieltä kuin suomea
    :param text: str, yksi puheenvuoro
    :return: str tai None, puheenvuoro tai None jos vierasta kieltä
    tunnistettiin olevan raja-arvoa suurempi osuus
    """
    # Raja-arvo
    threshold = 0.3

    words = text.split()

    if not words:
        return None

    # Poimitaan suomenkieliset sanat
    finnish_words = [word for word in words if is_finnish_word(word)]

    # Lasketaan muiden kuin suomenkielisten sanojen osuus kaikista sanoista
    non_finnish_ratio = 1 - (len(finnish_words) / len(words))

    # Jos vierasta kieltä tunnistettiin olevan raja-arvoa suurempi osuus, niin
    # palautetaan None
    if non_finnish_ratio > threshold:
        return None

    # Muutoin palautetaan alkuperäinen puheenvuoro
    return text


# Hyödynnetään functools.lru_cache-kirjastoa, joka tallentaa sanojen
# tarkistukset muistiin, jotta saataisiin kielen suodatus tehokkaammaksi
@lru_cache(maxsize=10000)
def is_finnish_word(word):
    """
    Tarkistaa, onko annettu sana suomenkielinen. Funktio käyttää
    cache-mekanismia nopeuttamaan käsittelyä
    :param word: str, Tarkistettava sana
    :return: bool, True, jos sana on suomenkielinen, False jos ei
    """
    try:
        return langdetect.detect(word) == 'fi'
    except langdetect.LangDetectException:
        return False


def parallel_processing(texts, function, chunksize=100):
    """
    Käsittelee listan sisältöä rinnakkaisesti annetulla funktiolla paremman
    tehokkuuden saavuttamiseksi.
    :param texts: list, 'clean_content' sarakkeen puheenvuorot listana
    :param function: Funktio, johon rinnakkaiskäsittelyä halutaan hyödyntää
    :param chunksize: int, kuinka monta puheenvuoroa jaetaan yhteen erään
    rinnakkaiskäsittelyssä. Oletusarvo on 100
    :return: list, joka sisältää funktion soveltamisen tulokset jokaiselle
    puheenvuorolle
    """
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(function, texts,
                                         chunksize=chunksize),
                            total=len(texts),
                            desc="Esikäsittely käynnissä"))
    return results


def lemmatize_content(text):
    """
    Funktio muuttaa puheenvuoron sanat perusmuotoon (lemmatisointi)
    Stanza-kirjaston avulla.
    :param text: str, yksi puheenvuoro
    :return: str, jossa sanat on muunnettu perusmuotoon
    """
    doc = nlp(text)
    return " ".join(
        [word.lemma for sentence in doc.sentences for word in sentence.words])


def stanza_processing(combined_df):
    """

    :param combined_df:
    :return:
    """
    parsed_docs = []
    for _, row in tqdm(combined_df.iterrows(), total=len(combined_df),
                       desc="Tokenisoidaan ja jäsennetään puheenvuoroja"):
        text = row['content']

        if isinstance(text, str) and text.strip():
            doc = nlp(text)  # Tokenisointi + jäsennys
        else:
            doc = None  # Puuttuva tai tyhjä teksti

        parsed_docs.append(doc)

    combined_df["parsed_doc"] = parsed_docs  # Tallennetaan Stanza-dokumentit
    return combined_df


def main_preprocessing(path_to_file, file, file_number, total_csv_files, df):
    """
    Funktio suorittaa esikäsittelyn tiedostolle: kielensuodatus,
    lemmatisointi, tekstin siivous, sarakkeet vuodelle sekä puhujan roolille
    :param path_to_file: str, tiedostopolku
    :param file: str, tiedoston nimi
    :param file_number: int, tiedoston numero
    :param total_csv_files: int, kaikkien tiedostojen lukumäärä
    :param df: DataFrame, tiedoston tiedot
    """
    # Tarkistetaan, onko esikäsittely jo tehty tarkistamalla löytyvätkö
    # lemmatisoidut puheenvuorot tiedostosta. Jos löytyvät, ei esikäsitellä
    # uudestaan
    if 'lemmatized_content' in df.columns:
        print(f"Tiedosto '{file}' valmis ({file_number}/{total_csv_files})")
        return

    # Korvataan lyhenne ed. edustajaksi ennen muuta käsittelyä, sillä se
    # vaikuttaa ominaisuuksien analysointiin
    df['content'] = df['content'].apply(replace_abbreviations)

    # Lisätään tiedostoon sarakkeet vuodelle ja puhujan roolille
    information = df[['date', 'speech']].to_numpy()
    information = date_and_speech_preprocessing(information)
    df['year'] = information[:, 0]
    df['role'] = information[:, 1]

    # Luodaan uusi sarake puheenvuoroille, joista on poistettu erikoismerkit ja
    # ylimääräiset välilyönnit.
    df['clean_content'] = df['content'].apply(clean_text)

    # Poistetaan ne rivit tiedostosta, jossa arvioidaan olevan liikaa vierasta
    # kieltä mukana.
    df = language_filtering_progress(df, 'clean_content')

    # Tehdään lemmatisointi puheenvuoroille ja lisätään lemmatisoidut tekstit
    # DataFrameen uutena sarakkeena
    if not df['clean_content'].empty:
        df['lemmatized_content'] = parallel_processing(
            df['clean_content'].tolist(), lemmatize_content)
    else:
        print("Tiedosto on tyhjä, ei esikäsiteltävää.")

    # Tallennetaan päivitetty DataFrame lähtötiedostoon
    df.to_csv(path_to_file, index=False, encoding='utf-8')
    print(f"Tiedosto '{file}' valmis ({file_number}/{total_csv_files})")
