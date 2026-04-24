#!/bin/python3
import itertools
import operator as op
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
import json
import os

import numpy as np
import pandas as pd
from utils.config import HISTOGRAM_COLOR_1, HISTOGRAM_EDGECOLOR_1, IMG_OUTPUT_PATH, AXIS_DESC_SIZE, AXIS_VALS_SIZE, TITLE_SIZE

from dataset.data_access import DatasetLoader


from dataset.data_access import DatasetLoader



def filter_by_number_of_places(number_of_places: int, data: Dict[str, list[object]], condition_cb: Callable[[int, int], bool] = op.eq):
    """
        Filters only users that rated specific number of restaurants.
    """

    return {k: v for k, v in data.items() if condition_cb(len(v), number_of_places)}

def get_topn_users_coverage(aggregated_authors_places: Dict[str, List[object]], min_ratings_count: int) -> Dict[str, List[object]]:

    aggregated_of_specific_size = filter_by_number_of_places(min_ratings_count, aggregated_authors_places, op.ge)

    unique_places = {}
    for key, value in list(aggregated_of_specific_size.items()):
        for place in value:
            place_id = place.get("id")
            unique_places[place_id] = value

    return unique_places

def get_places_with_reviews():
    """
        Returns places that has at least one review.
    """

    d_loader = DatasetLoader()
    places_data_raw = d_loader.load_data_raw()
    places_data = json_places_to_dictionary(places_data_raw)
    return get_places_data_with_review(places_data)

def get_places_data_with_review(places_data: Dict):
    return {k: v for k, v in places_data.items() if "reviews" in v and v['reviews'] != []}

def json_places_to_dictionary(raw_places_json_dict) -> Dict:
    """
    returns dictionary placeID => placeData for fast look up
    we expect ids to be unique!!
    """

    return { place['id']: place for place in raw_places_json_dict['places'] }

def process_place_review(place, review, authors_places) -> Dict:

    author_name = review['authorAttribution']['displayName']

    if author_name in authors_places:
        authors_places[author_name].append(place)
    else:
        authors_places[author_name] = [place]

def load_places_data():

    d_loader = DatasetLoader()
    places_data_raw = d_loader.load_data_raw()

    places_data = json_places_to_dictionary(places_data_raw)
    return places_data

def get_users_reviewed_places_dictionary(places_data: Dict[str, object]) -> Dict:
    """
    Returns dictionary mapping of user name to a list of places user voted for.
    """

    authors_places = {} # mapping of author => [] list of places ids
    for id, place in places_data.items():
        if 'reviews' in place:
            for review in place['reviews']:
                process_place_review(place, review, authors_places)

    print(len(authors_places))
    return authors_places

def aggregate_user_reviews_google_places_API() -> Dict[str, List[object]]:
    """
        Aggregates user reviews from the original places.json file.
        Returns a dictionary mapping of user name to a list of places user voted for.
    """

    places_data = load_places_data()
    user_reviewed_places = get_users_reviewed_places_dictionary(places_data)

    return user_reviewed_places


def get_users_reviews_count_frequencies(user_reviewed_places: Dict[str, List[object]]) -> List[int]:
    review_counts = [len(places) for places in user_reviewed_places.values()]
    return review_counts


def plot_user_places_coverage_histogram(frequencies, total_number_of_places = -1):

    reviewed_places_count = list(range(1, len(frequencies)+1))

    # Create the bar plot
    #plt.figure(figsize=(10, 5))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})

    cumulative_coverage = axes[0].bar(reviewed_places_count, frequencies, color='skyblue')

    # Customize the plot
    axes[0].set_xlabel('Number of Places Reviewed', fontsize=AXIS_DESC_SIZE)
    axes[0].set_ylabel('Number of Places Covered', fontsize=AXIS_DESC_SIZE)
    axes[0].set_title('Cumulative Histogram of Places Coverage', fontsize=TITLE_SIZE)
    axes[0].set_xticks(reviewed_places_count)

    axes[0].set_ylim(0, max(frequencies) + 200, fontsize=AXIS_VALS_SIZE)
    # Add labels on top of each bar
    for val in cumulative_coverage:
        height = val.get_height()
        axes[0].text(val.get_x() + val.get_width()/2, height + 0.1, str(height), ha='center', va='bottom', fontsize=10)

    frequencies_percentage = [round((freq / total_number_of_places) * 100, 2) for freq in frequencies]

    cumulative_coverage_percantages = axes[1].bar(reviewed_places_count, frequencies_percentage, color='red',width=0.3)


    # Customize the plot
    axes[1].set_xlabel('Number of Places Reviewed')
    axes[1].set_ylabel('Percentage of Places Covered')
    axes[1].set_xticks(reviewed_places_count)

    axes[1].set_ylim(0, max(frequencies_percentage) + 25)

    for val in cumulative_coverage_percantages:
        height = val.get_height()
        pos_x = val.get_x() + (val.get_width()/0.3)
        axes[1].text(pos_x, height + 0.1, str(height)+ "%", ha='center', va='bottom', fontsize=8, rotation=45)


    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "histogram-of-user-reviews-all.pdf"))
    plt.show()


def vykresli_histogram_pokryti_mist_s_pocty_lidi(frekvence, celkovy_pocet_mist=-1):
    """
    Vykreslí histogram absolutního pokrytí míst podle počtu ohodnocených míst uživateli.
    """
    if celkovy_pocet_mist <= 0:
        raise ValueError("Neplatný počet míst. Hodnota musí být větší než 0.")

    pocet_ohodnocenych = list(range(1, len(frekvence) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(pocet_ohodnocenych, frekvence, color=HISTOGRAM_COLOR_1, edgecolor=HISTOGRAM_EDGECOLOR_1)

    ax.set_xlabel('Počet ohodnocených míst uživatelem', fontsize=AXIS_DESC_SIZE)
    ax.set_ylabel('Počet míst v novém datasetu', fontsize=AXIS_DESC_SIZE)
    ax.set_title('Velikost datasetu omezeném na top-n uživatel', fontsize=TITLE_SIZE)
    ax.set_xticks(pocet_ohodnocenych)
    ax.tick_params(axis='x', labelsize=AXIS_VALS_SIZE)
    ax.set_ylim(0, max(frekvence) + 200)
    ax.tick_params(axis='y', labelsize=AXIS_VALS_SIZE)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, str(height),
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "user-place-coverage-histogram.pdf"))
    plt.show()

def vykresli_histogram_pokryti_mist(frekvence, celkovy_pocet_mist=-1):
    """
    Vykreslí histogram absolutního pokrytí míst podle počtu ohodnocených míst uživateli.
    """
    if celkovy_pocet_mist <= 0:
        raise ValueError("Neplatný počet míst. Hodnota musí být větší než 0.")

    pocet_ohodnocenych = list(range(1, len(frekvence) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(pocet_ohodnocenych, frekvence, color=HISTOGRAM_COLOR_1, edgecolor=HISTOGRAM_EDGECOLOR_1)

    ax.set_xlabel('Počet ohodnocených míst uživatelem', fontsize=AXIS_DESC_SIZE)
    ax.set_ylabel('Počet míst v novém datasetu', fontsize=AXIS_DESC_SIZE)
    ax.set_title('Velikost datasetu omezeném na top-n uživatel', fontsize=TITLE_SIZE)
    ax.set_xticks(pocet_ohodnocenych)
    ax.tick_params(axis='x', labelsize=AXIS_VALS_SIZE)
    ax.set_ylim(0, max(frekvence) + 200)
    ax.tick_params(axis='y', labelsize=AXIS_VALS_SIZE)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, str(height),
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "user-place-coverage-histogram.pdf"))
    plt.show()

def total_restaurants_coverage_analysis(aggregated_authors_places: Dict[str, List[object]], total_number_of_places = -1):

    res = []
    for min_place_count in range(1, 32):
        res.append(len(get_topn_users_coverage(aggregated_authors_places, min_place_count)))

    vykresli_histogram_pokryti_mist(res, total_number_of_places )


if __name__ == "__main__":
    aggregated_authors_places = aggregate_user_reviews_google_places_API()

    total_number_of_places = len(get_places_with_reviews())

    total_number_of_places = len(get_places_with_reviews())
    total_restaurants_coverage_analysis(aggregated_authors_places, total_number_of_places)
