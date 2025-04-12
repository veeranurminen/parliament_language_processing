"""
Tässä tiedostossa on syvällisempään kielen analyysiin keskittyviä
ominaisuuksien tutkimiseen tarvittavia funktioita.

Ominaisuudet:
    - Virkkeen juuresta riippuvien sanojen määrä eri vuosina
    - Virkkeen juuresta riippuvien sanojen määrä suhteessa virkkeen sanamäärään
      eri vuosina
    - Eri sanaluokkien osuudet puheenvuoroista eri vuosina
    - Sanaston monimuotoisuus eri vuosina type/token ration (TTR) avulla
    - Yhdyssanojen määrä eri vuosina
    - Substantiiviyhdyssanojen osuus puheenvuorojen kaikista substantiiveista
      eri vuosina
    - Alkuperäisten ja lemmatisoitujen sanojen pituussuhde eri vuosina

Tekijä: Veera Nurminen
2025 Tampere
"""

import stanza
import logging
import os
from tqdm import tqdm
import visualization_and_analysis
import random
import numpy as np
import sys


# Vähennetään Stanzan lokitietojen määrää
stanza.logger.setLevel(logging.WARNING)

# Ladataan Stanza-malli
if not os.path.exists(os.path.expanduser('~/.stanza_resources/fi')):
    stanza.download('fi')
nlp = stanza.Pipeline('fi', processors='tokenize,lemma,pos,mwt,depparse',
                      download_method=None)


def preprocess_and_store_tokens(combined_df):
    """
    Puheenvuorot käsitellään Stanzalla ja lisätään combined_df DataFrameen.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :return: Päivitetty DataFrame, johon on lisätty Stanza-analyysi
    """

    # Tokenisoidaan data
    processed_texts = []

    for _, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0],
                       desc="Tokenisoidaan puheenvuorot", file=sys.stdout):
        text = row['content']

        if not isinstance(text, str) or not text.strip():
            processed_texts.append(None)
            continue

        try:
            doc = nlp(text)
            processed_texts.append(doc)
        except Exception as e:
            print(f"Virhe tekstin käsittelyssä: {text[:50]}... ({e})")
            processed_texts.append(None)

    # Tarkistetaan, ettei pituus muutu
    assert len(processed_texts) == len(combined_df), "Rivimäärä ei täsmää!"

    # Lisätään tulokset DataFrameen
    combined_df["stanza_output"] = processed_texts

    return combined_df


def save_outliers_to_csv(df, column, threshold, output_file="outliers.csv"):
    """
    Funktio tallettaa csv-tiedostoon poikkeavia tietyn raja-arvon ylittäviä
    tuloksia aiheuttavia puheenvuoroja lähempää tarkastelua varten.

    :param df: DataFrame, tiedostojen tiedot
    :param column: str, sarake, josta poikkeavat arvot etsitään
    :param threshold: int, raja-arvo, jonka ylittävät arvot tallennetaan
    :param output_file: str, tiedostonimi, johon poikkeavat havainnot
    tallennetaan
    """
    # Suodatetaan poikkeavat arvot
    outliers = df[df[column] > threshold]

    # Tallennetaan valitut sarakkeet csv-tiedostoon
    outliers[['id', 'role', 'speaker', 'year', column, 'content',
              'lemmatized_content']].to_csv(output_file, index=True)


def count_root_dependencies(text):
    """
    Funktio laskee kuinka monta sanaa riippuu suoraan virkkeen juuresta.

    :param text: Stanza Sentence, tokenoitu virke
    :return: tuple (int, int), juuresta riippuvien sanojen lukumäärä ja sanojen
    kokonaismäärä
    """
    root_id = 0
    dependencies = 0

    # Etsitään virkkeen juuren id (head == 0)
    for word in text.words:
        if word.head == 0:
            root_id = word.id
            break

    # Lasketaan juuresta riippuvien sanojen lukumäärä tutkimalla
    # osoittavatko ne juuren indeksiin
    for word in text.words:
        if word.head == root_id:
            dependencies += 1

    return dependencies, len(text.words)


def root_dependencies(combined_df, sample_size_per_year=None):
    """
    Funktio laskee juuresta riippuvien sanojen määrän per virke eri vuosina.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param sample_size_per_year: int, määrittää analyysiin mukaan otettavien
    puheenvuorojen enimmäismäärän per vuosi
    :return: tuple (list, list), ominaisuuden arvot ja vuodet
    """
    print("\nAnalysoidaan virkkeen juuresta riippuvien sanojen määrää...")

    # Jos on annettu parametri sample_size_per_year, kutsutaan funktiota
    # sample_speeches, joka rajoittaa analysoitavien puheenvuorojen määrän
    # lukuun sample_size_per_year per vuosi ja satunnaistaa puheenvuorojen
    # valintaa
    if sample_size_per_year:
        combined_df = visualization_and_analysis.sample_speeches(
            combined_df, sample_size_per_year)

    years = []
    dependencies = []

    # Käydään läpi jokainen puheenvuoro
    for _, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0],
                       desc="Analysoidaan puheenvuoroja", file=sys.stdout):
        year = row['year']
        doc = row['stanza_output']

        if doc is None:
            continue

        # Käydään läpi jokainen puheenvuoron virke
        for sentence in doc.sentences:
            sentence_root_dependencies, _ = count_root_dependencies(sentence)
            years.append(int(year))
            dependencies.append(sentence_root_dependencies)

    return dependencies, years


def root_dependencies_divided_by_sentence_word_count(
        combined_df, sample_size_per_year=None):
    """
    Funktio laskee juuresta riippuvien sanojen määrän suhteessa virkkeen
    sanamäärään eri vuosina.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param sample_size_per_year: int, määrittää analyysiin mukaan otettavien
    puheenvuorojen enimmäismäärän per vuosi
    :return: tuple (list, list), ominaisuuden arvot ja vuodet
    """
    print("\nAnalysoidaan virkkeen juuresta riippuvien sanojen määrää suhteessa "
          "virkkeen sanamäärään...")

    # Jos on annettu parametri sample_size_per_year, kutsutaan funktiota
    # sample_speeches, joka rajoittaa analysoitavien puheenvuorojen määrän
    # lukuun sample_size_per_year per vuosi ja satunnaistaa puheenvuorojen
    # valintaa
    if sample_size_per_year:
        combined_df = visualization_and_analysis.sample_speeches(
            combined_df, sample_size_per_year)

    years = []
    dependencies = []

    # Käydään läpi jokainen puheenvuoro
    for _, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0],
                       desc="Analysoidaan puheenvuoroja", file=sys.stdout):
        year = row['year']
        doc = row['stanza_output']

        if doc is None:
            continue

        # Käydään läpi jokainen puheenvuoron virke
        for sentence in doc.sentences:
            sentence_root_dependencies, sentence_word_count = \
                count_root_dependencies(sentence)
            normalized_dependencies = sentence_root_dependencies / sentence_word_count \
                if sentence_word_count > 0 else 0

            years.append(int(year))
            dependencies.append(normalized_dependencies)

    return dependencies, years


def part_of_speech_analysis(combined_df, abbreviation, pos,
                            sample_size_per_year=None):
    """
    Funktio analysoi tietyn sanaluokan osuuden puheenvuoroissa eri vuosina.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param abbreviation: str, sanaluokan tai sanaluokkien UPOS tagit
    :param pos: str, sanaluokan nimen monikon genetiivi, esim. "Adjektiivien"
    :param sample_size_per_year: int, määrittää analyysiin mukaan otettavien
    puheenvuorojen enimmäismäärän per vuosi
    :return: tuple (list, list), ominaisuuden arvot ja vuodet
    """
    print(f"\nAnalysoidaan {pos.lower()} osuutta puheenvuoroissa...", flush=True)

    # Jos on annettu parametri sample_size_per_year, kutsutaan funktiota
    # sample_speeches, joka rajoittaa analysoitavien puheenvuorojen määrän
    # lukuun sample_size_per_year per vuosi ja satunnaistaa puheenvuorojen
    # valintaa
    if sample_size_per_year:
        combined_df = visualization_and_analysis.sample_speeches(
            combined_df, sample_size_per_year)

    years = []
    # Sanaluokan sanojen osuudet
    pos_ratios = []

    # Jos lyhenteitä on useampi välilyönnillä eroteltuna erotellaan ne listaksi
    abbreviation = abbreviation.split()

    for _, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0],
                       desc="Analysoidaan puheenvuoroja", file=sys.stdout):
        year = row['year']
        doc = row['stanza_output']

        if doc is None:
            continue

        total_words = sum(len(sentence.words) for sentence in doc.sentences)
        # Kyseisen sanaluokan sanojen määrä
        pos_count = sum(1 for sentence in doc.sentences for word in
                        sentence.words if word.upos in abbreviation)

        if total_words > 0:
            pos_proportion = pos_count / total_words
            years.append(int(year))
            pos_ratios.append(pos_proportion)

    return pos_ratios, years


def calculate_ttr(text, sample_size=1000):
    """
    Funktio laskee type/token ration (TTR) annetulle tekstille.

    :param text: str, yhden vuoden puheenvuorot yhtenä merkkijonona
    :param sample_size: int, otoksen koko, eli kuinka monta peräkkäistä sanaa
    otetaan kerrallaan analyysiin
    :return: float tai None, lasketun otoksen TTR-luku tai None, jos sanoja on
    liian vähän
    """
    # Jaetaan teksti sanoiksi
    all_words = text.split()

    # Tarkistetaan, että teksti sisältää tarpeeksi sanoja analyysiin
    if len(all_words) < sample_size:
        print(f"Virhe: liian pieni otos, tarvitaan vähintään {sample_size} "
              f"sanaa jokaiselta vuodelta")
        return None

    # Valitaan satunnainen aloituskohta ja otetaan sample_size verran
    # peräkkäisiä sanoja
    start_index = random.randint(0, len(all_words) - sample_size)
    sample = all_words[start_index:start_index + sample_size]

    # Lasketaan uniikkien sanojen määrä otoksessa
    unique_words = set(sample)

    # Lasketaan type/token ratio (TTR)
    ttr = len(unique_words) / sample_size
    return ttr


def ttr_analysis(combined_df, sample_size=1000, number_of_samples=1000):
    """
    Funktio analysoi sanaston monimuotoisuutta type/token ration (TTR) avulla.
    TTR-luku lasketaan sample_size ilmaisemalle määrälle peräkkäisiä sanoja, ja
    luku lasketaan number_of_samples kertaa jokaista vuotta kohden.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param sample_size: int, analyysissä käytettävien peräkkäisten sanojen
    määrä yhtä otosta kohden
    :param number_of_samples: int, kuinka monta otosta otetaan vuotta kohden
    :return: tuple (list, list), ominaisuuden arvot ja vuodet
    """
    print("\nSuoritetaan TTR-analyysiä...")

    years = []
    ttr_values = []

    # Ryhmitellään puheenvuorot vuoden perusteella
    grouped = combined_df.groupby("year")

    # Käydään puheenvuorot läpi vuosi kerrallaan
    for year, group in tqdm(grouped,
                            desc="Analysoidaan TTR-lukuja vuosi kerrallaan",
                            file=sys.stdout):
        # Yhdistetään kaikki vuoden puheenvuorot yhdeksi merkkijonoksi
        speeches_as_text = " ".join(group["content"].dropna())

        # Kaikkien vuoden sanojen määrä yhteensä
        word_count = len(speeches_as_text.split())

        # Ohitetaan vuodet, joissa on liian vähän sanoja
        if word_count < sample_size:
            continue

        # Määritetään TTR useilla otoksilla
        for _ in range(number_of_samples):
            ttr = calculate_ttr(speeches_as_text, sample_size)
            if ttr is not None:
                years.append(year)
                ttr_values.append(ttr)

    return ttr_values, years


def compound_word_analysis(combined_df, sample_size_per_year):
    """
    Funktio analysoi yhdyssanojen osuuden puheenvuoroa kohden eri vuosina.
    Yhdyssanat lasketaan laskemalla sanat joissa on merkki "#", sillä
    esikäsitellyissä puheenvuoroissa Stanza merkkaa yhdyssanoissa sanojen
    väliin merkin "#".

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param sample_size_per_year: int, määrittää analyysiin mukaan otettavien
    puheenvuorojen enimmäismäärän per vuosi
    :return: tuple (pandas Series, pandas Series), jokaiselle puheenvuorolle
    lasketut yhdyssanaosuudet ja vastaavat vuodet
    """
    # Jos on annettu parametri sample_size_per_year, kutsutaan funktiota
    # sample_speeches, joka rajoittaa analysoitavien puheenvuorojen määrän
    # lukuun sample_size_per_year per vuosi ja satunnaistaa puheenvuorojen
    # valintaa
    if sample_size_per_year:
        combined_df = visualization_and_analysis.sample_speeches(
            combined_df, sample_size_per_year)

    # Poistetaan rivit, joissa puuttuu joko vuosi tai sisältö
    filtered_df = combined_df.dropna(
        subset=['lemmatized_content', 'year']).copy()

    # Lasketaan jokaiselle puheenvuorolle yhdyssanojen osuus
    filtered_df['compound_ratio'] = filtered_df['lemmatized_content'].apply(
        lambda text: sum(1 for word in text.split() if "#" in word) / len(
            text.split())
        if isinstance(text, str) and len(text.split()) > 0 else None
    )

    return filtered_df['compound_ratio'], filtered_df['year']


def noun_compound_analysis(combined_df, sample_size_per_year):
    """
    Funktio analysoi substantiiviyhdyssanojen osuuden kaikista substantiiveista
    eri vuosina. Substantiiviyhdyssanat lasketaan laskemalla substantiivit
    joissa on merkki "#", sillä esikäsitellyissä puheenvuoroissa Stanza merkkaa
    yhdyssanoissa sanojen väliin merkin "#".

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param sample_size_per_year: int, määrittää analyysiin mukaan otettavien
    puheenvuorojen enimmäismäärän per vuosi
    :return: tuple (list, list), ominaisuuden arvot ja vuodet
    """
    print("\nAnalysoidaan substantiiviyhdyssanojen osuutta "
          "kaikista substantiiveista...")

    # Jos on annettu parametri sample_size_per_year, kutsutaan funktiota
    # sample_speeches, joka rajoittaa analysoitavien puheenvuorojen määrän
    # lukuun sample_size_per_year per vuosi ja satunnaistaa puheenvuorojen
    # valintaa
    if sample_size_per_year:
        combined_df = visualization_and_analysis.sample_speeches(
            combined_df, sample_size_per_year)

    years = []
    noun_compound_ratios = []

    # Käydään läpi jokainen puheenvuoro
    for _, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0],
                       desc="Analysoidaan puheenvuoroja", file=sys.stdout):
        year = row['year']
        doc = row['stanza_output']

        if doc is None:
            continue

        try:
            # Substantiivien määrä
            total_nouns = 0
            # Substantiiviyhdyssanojen määrä
            noun_compound_count = 0

            for sentence in doc.sentences:
                for word in sentence.words:
                    if word.upos == "NOUN":
                        total_nouns += 1
                        if "#" in word.lemma:
                            noun_compound_count += 1

            if total_nouns > 0:
                ratio = noun_compound_count / total_nouns
                years.append(int(year))
                noun_compound_ratios.append(ratio)

        except Exception as e:
            print(f"Virhe tekstin käsittelyssä: {str(e)}")
            years.append(0)
            noun_compound_ratios.append(0)
            continue

    return noun_compound_ratios, years


def word_vs_lemmatized_word_ratio(orig_text, lemm_text):
    """
    Laskee keskimääräisen pituussuhteen alkuperäisten sanojen ja niiden
    lemmatisoitujen vastineiden välillä. Puheenvuorot, joissa alkuperäisten
    sanojen ja vastaavien lemmatisoitujen sanojen määrät eivät täsmää
    sivuutetaan.

    :param orig_text: str, alkuperäinen puheenvuoro
    :param lemm_text: str, lemmatisoitu versio puheenvuorosta
    :return: float tai None, keskimääräinen pituussuhde tai None, jos ei
    laskettavissa
    """
    # Poistetaan lemmatisoinnin yhteydessä lisätyt "#"-merkit
    lemm_text = lemm_text.replace("#", "")

    # Jaetaan tekstit sanoiksi
    orig_words = orig_text.split()
    lemm_words = lemm_text.split()

    ratios = []

    # Jos alkuperäisessä ja lemmatisoidussa puheenvuorossa ei ole yhtä paljon
    # sanoja, puheenvuoro ohitetaan
    if len(orig_words) != len(lemm_words):
        return None

    # Lasketaan sanojen pituuksien suhde:
    # (alkup. sanan merkkimäärä) / (lemm. sanan merkkimäärä)
    # Suhde on None, jos lemmatisoidussa sanassa ei ole yhtäkään merkkiä
    for i in range(len(orig_words)):
        ratio = (len(orig_words[i]) / len(lemm_words[i])) \
            if len(lemm_words[i]) > 0 else None
        ratios.append(ratio)

    # Suodatetaan pois mahdolliset None-arvot
    ratios = [ratio for ratio in ratios if ratio is not None]

    # Palautetaan suhteiden keskiarvo
    return np.mean(ratios) if ratios else None


def word_vs_lemmatized_word_analysis(combined_df, sample_size_per_year=None):
    """
    Funktio analysoi alkuperäisten sanojen ja niiden lemmatisoitujen
    vastineiden pituussuhdetta eri vuosina. Pituussuhde lasketaan jokaiselle
    sanaparille, ja jokaiselle puheenvuorolle määritetään pituussuhteiden
    keskiarvo. Datapisteitä tulee siis yksi jokaista puheenvuoroa kohden,
    mutta puheenvuorot joissa alkuperäisten sanojen ja vastaavien
    lemmatisoitujen sanojen määrät eivät täsmää sivuutetaan.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param sample_size_per_year: int, määrittää analyysiin mukaan otettavien
    puheenvuorojen enimmäismäärän per vuosi
    :return: tuple (list, list), ominaisuuden arvot ja vuodet
    """
    print("\nAnalysoidaan alkuperäisten sanojen pituutta verrattuna niiden "
          "lemmatisoituihin vastineisiin...")

    # Jos on annettu parametri sample_size_per_year, kutsutaan funktiota
    # sample_speeches, joka rajoittaa analysoitavien puheenvuorojen määrän
    # lukuun sample_size_per_year per vuosi ja satunnaistaa puheenvuorojen
    # valintaa
    if sample_size_per_year:
        combined_df = visualization_and_analysis.\
            sample_speeches(combined_df, sample_size_per_year)

    years = []
    ratios = []

    # Käydään läpi jokainen puheenvuoro
    for _, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0],
                       desc="Analysoidaan puheenvuoroja", file=sys.stdout):
        year = row['year']
        orig_clean_content = row['clean_content']
        lemm_content = row['lemmatized_content']

        # Ohitetaan tyhjät tai puuttuvat tekstit
        if not isinstance(orig_clean_content, str) or not isinstance(
                lemm_content, str):
            continue

        # Lasketaan suhdeluku
        ratio = word_vs_lemmatized_word_ratio(orig_clean_content, lemm_content)

        if ratio is not None:
            years.append(int(year))
            ratios.append(ratio)

    return ratios, years
