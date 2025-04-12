"""
Tässä tiedostossa on yleisiin puheen piirteisiin liittyviä ominaisuuksien
analysointiin tarvittavia funktioita.

Ominaisuudet:
    - Puheenvuorojen pituus eri vuosina
    - Sanojen keskimääräinen merkkimäärä puheenvuoroa kohden eri vuosina

Tekijä: Veera Nurminen
2025 Tampere
"""

import pandas as pd
import visualization_and_analysis


def analyze_feature_over_years(combined_df, feature,
                               sample_size=None, threshold=None):
    """
    Funktio analysoi annetun ominaisuuden kehitystä eri vuosina.
    Funktio suodattaa pois tietyn raja-arvon ylittävät arvot, jos raja-arvo on
    annettu.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param feature: str, analysoitavan ominaisuuden DataFrame-sarake
    :param sample_size: int, määrittää analyysiin mukaan otettavien
    puheenvuorojen enimmäismäärän per vuosi
    :param threshold: int, jos raja-arvo on annettu, sitä suuremmat
    ominaisuuden arvot suodatetaan pois
    :return: tuple (NumPy-array, NumPy-array), ominaisuuden arvot ja vuodet
    """
    # Jos on annettu parametri sample_size, kutsutaan funktiota
    # sample_speeches, joka rajoittaa analysoitavien puheenvuorojen määrän
    # lukuun sample_size per vuosi ja satunnaistaa puheenvuorojen valintaa
    if sample_size:
        combined_df = visualization_and_analysis.sample_speeches(combined_df,
                                                                 sample_size)

    # Poistetaan rivit, joissa puuttuu joko ominaisuus tai vuosi
    df = combined_df.dropna(subset=[feature, 'year']).copy()

    # Suodatetaan raja-arvoa suuremmat rivit pois, jos threshold on annettu
    if threshold:
        df = df[df[feature] <= threshold]

    # Muutetaan vuodet ja ominaisuuksien arvot numeerisiksi
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

    # Muutetaan arvot NumPy-taulukoiksi
    years = df['year'].values
    values = df[feature].values

    return values, years


def speech_length_analysis(combined_df, sample_size_per_year=None):
    """
    Funktio analysoi puheenvuorojen pituudet merkkeinä eri vuosina.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param sample_size_per_year: int, määrittää analyysiin mukaan
    otettavien puheenvuorojen enimmäismäärän per vuosi
    :return: tuple (NumPy-array, NumPy-array), ominaisuuden arvot ja vuodet
    """
    print("Analysoidaan puheenvuorojen pituuksia...")

    # Lasketaan puheenvuoron pituus merkkeinä
    combined_df['speech_length'] = \
        combined_df['clean_content'].apply(lambda text_data:
                                           len(str(text_data)))

    values, years = analyze_feature_over_years(
        combined_df, 'speech_length', sample_size_per_year, threshold=60000)

    return values, years


def word_length_analysis(combined_df, sample_size_per_year=None):
    """
    Funktio analysoi puheenvuorojen keskimääräistä sanan pituutta eri vuosina.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param sample_size_per_year: int, määrittää analyysiin mukaan
    otettavien puheenvuorojen enimmäismäärän per vuosi
    :return: tuple (NumPy-array, NumPy-array), ominaisuuden arvot ja vuodet
    """
    print("Analysoidaan puheenvuorojen keskimääräistä sananpituutta...")

    # Lasketaan sanojen keskipituus, käsitellään vain validit rivit
    combined_df['word_length'] = combined_df['clean_content'].apply(
        lambda text: sum(len(word) for word in
                         text.split()) / len(text.split())
        if isinstance(text, str) and len(text.split()) > 0 else None
    )

    values, years = analyze_feature_over_years(
        combined_df, 'word_length', sample_size_per_year)

    return values, years
