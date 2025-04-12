"""
Tätä ohjelmaa voidaan hyödyntää tutkittaessa ajan myötä tapahtuvia
kielenmuutoksia. Ohjelma on osa kandidaatintyötäni "Eduskunnan täysistuntojen
puheenvuorojen kielenkäytön muutokset 1980-luvulta nykypäivään".

Ohjelmassa on käytetty aineistona Parlamenttisampo.fi portaalista ladattuja
csv-tiedostoja, jotka sisältävät tietoa eduskunnan puheenvuoroista ja
puhujista eri vuosilta. Tiedostoista talletetaan tietoa eduskunnan
puheenvuoroista tietorakenteeseen ja esikäsitellään tietoja muun muassa
lemmatisoimalla puheenvuoroja ja suodattamalla muita kuin suomenkielisiä
puheenvuoroja pois. Stanza-kirjaston avulla käsitellyt puheenvuorot
tallennetaan hdf-tiedostoon. Vuosikohtaisten tietojen pohjalta tutkitaan kielen
eri ominaisuuksien muutoksia, ja tulokset tallennetaan erilliseen
npz-tiedostoon. Lopulta tulokset visualisoidaan.

Tutkittuja ominaisuuksia:
    - Puheenvuorojen pituus eri vuosina
    - Sanojen keskimääräinen merkkimäärä puheenvuoroa kohden eri vuosina
    - Virkkeen juuresta riippuvien sanojen määrä eri vuosina
    - Virkkeen juuresta riippuvien sanojen määrä suhteessa virkkeen sanamäärään
      eri vuosina
    - Sanaluokkien osuuksia puheenvuoroista eri vuosina
    - Sanaston monimuotoisuutta eri vuosina type/token ration (TTR) avulla
    - Yhdyssanojen määrä eri vuosina
    - Substantiiviyhdyssanojen osuus puheenvuorojen kaikista substantiiveista
      eri vuosina
    - Alkuperäisten ja lemmatisoitujen sanojen pituussuhde eri vuosina

Tekijä: Veera Nurminen
2025 Tampere
"""

import pandas as pd
import os
import logging
import preprocessing
import feature_analysis
import complexity_analysis
import visualization_and_analysis
import numpy as np
import matplotlib.pyplot as plt


# Vähennetään yleisten Python-lokien määrää
logging.getLogger("stanza").setLevel(logging.WARNING)

feature_data = {}


def loading_data(folder):
    """
    Funktio lataa tiedot .csv-tiedostoista ja suorittaa tiedoille esikäsittelyn
    kutsumalla esikäsittelyfunktiota main_preprocessing.

    :param folder: str, tiedoston nimi
    :return: DataFrame, kaikki tiedostojen data, johon on päivitetty lisätietoa
    esikäsittelyn yhteydessä
    """
    # Tarkistetaan, että annettu kansio löytyy
    if not os.path.exists(folder):
        print(f"Virhe: Kansiota '{folder}' ei löytynyt.")
        # Lopetetaan suoritus
        return

    # Lasketaan käsiteltävien tiedostojen kokonaismäärä
    total_csv_files = len(
        [file for file in os.listdir(folder) if file.endswith(".csv")])

    combined_df = pd.DataFrame()

    # Käydään läpi kaikki kansion CSV-tiedostot
    file_number = 1
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path_to_file = os.path.join(folder, file)

            try:
                # Yritetään ladata tiedosto DataFrameen
                df = pd.read_csv(path_to_file)
            except Exception as e:
                print(f"Virhe tiedoston {file} lukemisessa: {e}")
                continue

            # Esikäsitellään tiedosto
            preprocessing.main_preprocessing(path_to_file, file, file_number,
                                             total_csv_files, df)

            # Tallennetaan päivitetty tiedosto
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            file_number += 1

    print(f"Kaikki tiedostot esikäsitelty")
    print()
    return combined_df


# All word classes that could be analyzed:
# adjective, adposition, adverb, auxiliary, coordinating conjunction,
# determiner, interjection, noun, numeral, particle, pronoun, proper noun,
# punctuation, subordinating conjunction, symbol, verb, other
word_classes = ["ADJ", "ADP", "ADV", "AUX", "CCONJ",
                "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
                "PUNCT", "SCONJ", "SYM", "VERB", "X"]

# Analyysiin valitut sanaluokat. Sanakirja, jossa avaimena on sanaluokan
# lyhenne (UPOS tagi), ja arvona sanaluokan nimen monikon genetiivi, jota
# hyödynnetään tuloksia visualisoidessa. Jos sanaluokkia halutaan samaan
# analyysiin useampia, niiden lyhenteet erotetaan välilyönnillä
POS_analysis = {
                "VERB": "Verbien",
                "NOUN": "Substantiivien",
                "PRON": "Pronominien",
                "CCONJ SCONJ": "Konjunktioiden"
                }

"""ADJ": "Adjektiivien", CCONJ": "Rinnastuskonjunktioiden", "SCONJ": "Alistuskonjunktioiden,
                "INTJ": "Interjektioiden"""


def save_information(file_name, feature_name, values, years):
    """

    :param file_name:
    :param feature_name:
    :param values:
    :param years:
    :return:
    """
    try:
        # Ladataan olemassa olevat tiedot, jos tiedosto on jo olemassa
        try:
            existing_data = np.load(file_name, allow_pickle=True)
            data_dict = dict(existing_data)
        except FileNotFoundError:
            data_dict = {}

        # Päivitetään sanakirja uusilla tiedoilla
        data_dict[feature_name] = values
        data_dict[f"{feature_name}_year"] = years

        # Tallennetaan päivitetty data takaisin tiedostoon
        np.savez(file_name, **data_dict)
    except Exception as e:
        print(f"Virhe tallennettaessa ominaisuutta {feature_name}: {e}")


def feature_extraction(combined_df):
    """
    Funktio, josta kutsutaan haluttuja ominaisuuksien analysointi -funktioita.
    Analyysien tulokset tallennetaan feature_data-sanakirjaan.

    :param combined_df: DataFrame, kaikki tiedostojen data
    """
    file_name = "language_metrics.npz"
    # Muuttujan sample_size_per_year avulla voidaan määrittää kuinka monta
    # puheenvuoroa jokaiselta vuodelta otetaan mukaan analyysiin. Puheenvuorot
    # valitaan satunnaisesti. Jos vuodelta on vähemmän puheenvuoroja saatavilla
    # kuin sample_size_per_year, otetaan  kaikki saatavat puheenvuorot mukaan
    # analyysiin.
    sample_size_per_year = None

    # Puheenvuorojen pituus eri vuosina
    values, years = feature_analysis.speech_length_analysis(
        combined_df, sample_size_per_year)
    save_information(file_name, "speech_length", values, years)

    # Sanojen keskimääräinen merkkimäärä puheenvuoroa kohden eri vuosina
    values, years = feature_analysis.word_length_analysis(
        combined_df, sample_size_per_year)
    save_information(file_name, "word_length", values, years)

    combined_df = complexity_analysis.preprocess_and_store_tokens(combined_df)

    # Virkkeen juuresta riippuvien sanojen määrä eri vuosina
    values, years = complexity_analysis.root_dependencies(
        combined_df, sample_size_per_year)
    save_information(file_name, "root_dependencies", values, years)

    # Virkkeen juuresta riippuvien sanojen määrä suhteessa virkkeen sanamäärään
    # eri vuosina
    values, years = complexity_analysis.\
        root_dependencies_divided_by_sentence_word_count(combined_df,
                                                         sample_size_per_year)
    save_information(file_name, "root_dependencies_divided", values, years)

    # Sanaluokkien osuudet puheenvuoroista eri vuosina
    for key, value in POS_analysis.items():
        values, years = complexity_analysis.part_of_speech_analysis(
            combined_df, key, value, sample_size_per_year)
        save_information(file_name, key, values, years)

    # Sanaston monimuotoisuus eri vuosina type/token ration (TTR) avulla.
    # Funktiossa on oletusarvona 1000 peräkkäisen sanan analysointi 1000 kertaa
    # vuotta kohden
    values, years = complexity_analysis.ttr_analysis(combined_df)
    save_information(file_name, "TTR", values, years)

    # Yhdyssanojen määrä eri vuosina
    values, years = complexity_analysis.compound_word_analysis(
        combined_df, sample_size_per_year)
    save_information(file_name, "compound_words", values, years)
    # Substantiiviyhdyssanojen osuus puheenvuorojen kaikista substantiiveista
    # eri vuosina
    values, years = complexity_analysis.noun_compound_analysis(
        combined_df, sample_size_per_year)
    save_information(file_name, "noun_compound_words", values, years)

    # Alkuperäisten ja lemmatisoitujen sanojen pituussuhde eri vuosina
    values, years = complexity_analysis.word_vs_lemmatized_word_analysis(
        combined_df, sample_size_per_year)
    save_information(file_name, "length_ratios", values, years)


def plot():
    """
    Funktio visualisoi jokaisen tutkitun ominaisuuden tulokset. Kuvaajia
    laitetaan kaksi vierekkäin samaan kuvaan, poikkeuksen sanaluokkien
    kuvaajat, jotka tulevat kaikki samaan kuvaan (2x4).
    """
    filename = "language_metrics.npz"
    data = np.load(filename)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    visualization_and_analysis.correlation_and_plot(
        data['speech_length_year'],
        data['speech_length'],
        "Puheenvuoron pituus merkkeinä",
        "Puheenvuorojen pituus eri vuosina", yticks=np.arange(0, 60001, 5000),
        ax=axes[0])
    visualization_and_analysis.correlation_and_plot(
        data['word_length_year'],
        data['word_length'],
        "Sanojen keskipituus",
        "Sanojen keskipituus per puheenvuoro eri vuosina",
        ax=axes[1])
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    visualization_and_analysis.correlation_and_plot(
        data['root_dependencies_year'],
        data['root_dependencies'],
        "Juuresta riippuvien sanojen määrä",
        "Juuresta riippuvien sanojen määrä virkettä\n kohden eri vuosina",
        ax=axes[0])
    visualization_and_analysis.correlation_and_plot(
        data['root_dependencies_divided_year'],
        data['root_dependencies_divided'],
        "Juuresta riippuvien sanojen määrä\n jaettuna virkkeen sanamäärällä",
        "Juuresta riippuvien sanojen määrä virkettä kohden\n eri vuosina "
        "suhteessa virkkeen sanamäärään",
        ax=axes[1])
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    i = 0
    for key, value in POS_analysis.items():
        row, col = divmod(i, 2)
        if key == "INTJ":
            y_limit = (0, 0.02)
        else:
            y_limit = None
        visualization_and_analysis.correlation_and_plot(
            data[f'{key}_year'],
            data[key],
            f"{value} osuus \npuheenvuorossa",
            f"{value} osuus puheenvuoroissa eri vuosina",
            y_limit=y_limit, ax=axes[row, col])
        i += 1
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    visualization_and_analysis.correlation_and_plot(
        data['TTR_year'],
        data['TTR'],
        "Type/Token Ratio",
        "Type/Token Ratio eri vuosina", y_limit=(0.3, 0.9),
        ax=axes[0])
    visualization_and_analysis.correlation_and_plot(
        data['length_ratios_year'],
        data['length_ratios'],
        "Keskimääräinen pituussuhde",
        "Alkuperäisten ja lemmatisoitujen sanojen\n pituussuhde eri vuosina",
        ax=axes[1])
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    visualization_and_analysis.correlation_and_plot(
        data['compound_words_year'],
        data['compound_words'],
        "Yhdyssanojen osuus per puheenvuoro",
        "Yhdyssanojen osuus per puheenvuoro eri vuosina",
        ax=axes[0])
    visualization_and_analysis.correlation_and_plot(
        data['noun_compound_words_year'],
        data['noun_compound_words'],
        "Substantiiviyhdyssanojen osuus \npuheenvuoron substantiiveista",
        "Substantiiviyhdyssanojen osuus \npuheenvuorojen substantiiveista "
        "eri vuosina",
        ax=axes[1])
    plt.tight_layout()
    plt.show()


def main():
    # Valitaan kansio, josta tiedostot löytyvät
    folder = "eduskunta_puheet"
    # folder = "testi_suom"

    # Ladataan data tietorakenteeseen
    combined_df = loading_data(folder)

    # Analysoidaan ominaisuuksia
    # feature_extraction(combined_df)

    # Tulosten visualisointi
    plot()


if __name__ == "__main__":
    main()
