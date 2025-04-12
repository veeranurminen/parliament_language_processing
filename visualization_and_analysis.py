"""
Tässä tiedostossa on yleistoimintoja, joita hyödynnetään useissa ohjelman
vaiheissa. Toimintoja:
    - Analysoitavien puheenvuorojen määrän rajoittaminen ja puheenvuorojen
      satunnaistaminen sample_speeches-funktion avulla
    - Lineaarinen regressioanalyysi
    - Pearsonin korrelaatio
    - Tulosten visualisointi scatter plotilla

Tekijä: Veera Nurminen
2025 Tampere
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def sample_speeches(combined_df, sample_size):
    """
    Funktio valitsee satunnaisesti sample_size määrän puheenvuoroja
    jokaiselta vuodelta. Jos tietyn vuoden puheenvuorojen määrä on pienempi
    kuin sample_size, otetaan kaikki saatavilla olevat puheenvuorot
    kyseiseltä vuodelta.

    :param combined_df: DataFrame, kaikki tiedostojen data
    :param sample_size: int, määrittää analyysiin mukaan otettavien
    puheenvuorojen enimmäismäärän per vuosi
    :return: DataFrame, sisältää satunnaisesti valitut rivit jokaiselta
    vuodelta
    """
    # Poistetaan rivit, joissa on NaN-arvoja
    combined_df = combined_df.dropna(subset=["year"])

    # Ryhmitellään vuoden mukaan ja otetaan satunnaisotanta jokaiselta vuodelta
    new_df = combined_df.groupby("year").apply(
        lambda all_speeches_of_a_year: all_speeches_of_a_year.sample(
            n=min(sample_size, len(all_speeches_of_a_year)),
            random_state=42))

    # Palautetaan uusi DataFrame, poistetaan monitasoinen indeksi ja luodaan
    # yksinkertainen indeksi
    return new_df.reset_index(drop=True)


def compute_linear_regression(x, y):
    """
    Suorittaa lineaarisen regression x- ja y-arvoille.

    :param x: NumPy-array, selittävä muuttuja (vuodet)
    :param y: NumPy-array, selitettävä muuttuja (analysoitavaan ominaisuuteen
    liittyvät arvot)
    :return: NumPy-array, mallin ennustamat y-arvot
    """
    model = LinearRegression()
    model.fit(x, y)
    return model.predict(x)


def pearson_correlation(years, values):
    """
    Määrittää Pearsonin korrelaation mukaisen korrelaatiokertoimen ja p-arvon.
    Jos p-arvo on yli 0.05, sen arvoksi merkataan > 0.05. Jos se on alle
    0.001, sen arvoksi merkataan < 0.001. Muutoin säilytetään sen arvo kolmen
    desimaalin tarkkuudella.

    :param years: NymPy-array, vuodet
    :param values: NymPy-array, analysoitavaan ominaisuuteen liittyvät arvot
    :return: tuple (float, str), korrelaatiokerroin ja merkkijono, jolla
    ilmaistaan p-arvo
    """
    # Pearsonin korrelaatio
    r, p = pearsonr(years, values)
    if 0.05 >= p >= 0.001:
        approx_p_value = f"p = {p:.3f}"
    elif p > 0.05:
        approx_p_value = f"p > 0.05"
    else:
        approx_p_value = f"p < 0.001"

    return r, approx_p_value


# Suurennetaan fonttikokoa
plt.rcParams.update({'font.size': 18})


def correlation_and_plot(years, values, ylabel, title, yticks=None, y_limit=None, ax=None):
    """
    Funktiossa visualisoidaan tuloksia scatter plotilla, johon on lisätty
    hieman kohinaa x-akselille. Kuvaan sovitetaan suora lineaariregressiolla.
    Määritetään myös Pearson korrelaatio ominaisuuden ja vuoden välillä.

    :param years: list, vuodet
    :param values: list, analysoitavan ominaisuuden arvot
    :param ylabel: str, y-akselin teksti
    :param title: str, kuvaajan otsikko
    :param yticks:
    :param y_limit:
    :param ax: matplotlibin Axes-olio, määrittää mihin kuvaaja piirretään
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Muutetaan tiedot NumPy-taulukoiksi ja järjestetään vuosien mukaan
    years = np.array(years)
    values = np.array(values)

    # Järjestetään tiedot nousevaan järjestykseen vuosien mukaan
    sorted_indices = np.argsort(years)
    years, values = years[sorted_indices], values[sorted_indices]

    # Pearson korrelaatio
    r, p = pearson_correlation(years, values)

    # Lineaarinen regressio
    x = years.reshape(-1, 1)
    y = values
    y_pred = compute_linear_regression(x, y)

    # Lasketaan vuosikohtaiset keskiarvot
    df = pd.DataFrame({"year": years, "value": values})
    yearly_means = df.groupby("year")["value"].mean()

    # Piirretään kuva sekä sovitetaan suora
    # Piirretään violin plot
    unique_years = sorted(set(years))
    data_per_year = [values[years == year] for year in unique_years]
    violin_parts = ax.violinplot(data_per_year, positions=unique_years,
                                 widths=1.2, bw_method=0.5)
    if yticks is not None:
        ax.set_yticks(yticks)

    if y_limit is not None:
        ax.set_ylim(y_limit)

    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    ax.plot(years, y_pred, color="red", label="Sovitettu suora")

    # Näytetään vuosikohtaiset keskiarvot kuvaajassa
    ax.scatter(yearly_means.index, yearly_means.values, color="red",
               marker="o", label="Vuosikohtainen keskiarvo")

    ax.set_xlabel("Vuosi")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='upper right', edgecolor="0", fontsize=14)

    # Lisätään korrelaatiokerroin ja p-arvo näkyviin kuvaan
    text = f'r = {r:.3f}\n{p}'
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=18,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
