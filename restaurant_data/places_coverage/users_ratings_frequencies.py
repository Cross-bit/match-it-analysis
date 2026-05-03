#!/bin/python3
import itertools
import operator as op
from typing import Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
import json
import os

import numpy as np
import pandas as pd
from utils.config import (
    HISTOGRAM_COLOR_1,
    HISTOGRAM_EDGECOLOR_1,
    IMG_OUTPUT_PATH,
    RESTAURANT_DATASET_ROOT,
)
from dataset.data_access import DatasetLoader



# ===============================================================
# DESCRIPTION
# ===============================================================
# This script contains analysis methods over the data obtained
# from the (new) places API.
#
# Plots are exported into the ./imgs directory in .pdf format.
#


PLACES_DATA_FILE = str(RESTAURANT_DATASET_ROOT / "places.json")


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


def get_users_reviews_count_frequencies(user_reviewed_places: Dict[str, List[object]]) -> List[int]:
    review_counts = [len(places) for places in user_reviewed_places.values()]
    return review_counts


# ===============================================================
### Plotting frequency of user reviews and other data analysis
# ===============================================================

def plot_users_reviews_count_histogram(reviews_frequencies: List[int], max_samples=15):
    aggregated_frequencies = [0] * (max_samples + 1)
    should_cumulate_rest = max_samples < max(reviews_frequencies)
    print(sum(reviews_frequencies))

    for review in reviews_frequencies:
        if review <= max_samples or not should_cumulate_rest:
            aggregated_frequencies[review] += 1
        else:
            aggregated_frequencies[max_samples] += 1

    # Vytvoření X pozic bez 0 – tím odstraníme levé odsazení
    x = np.arange(1, max_samples + 1) * 2.2  # rovnoměrné rozestupy

    # Připravíme popisky osy X
    x_labels = [str(i) for i in range(1, max_samples if should_cumulate_rest else max_samples + 1)]
    if should_cumulate_rest:
        x_labels.append(f'>{max_samples}')
        x = np.append(x, x[-1] + 2.2)  # přidáme extra pozici pro ">max"

    # Vykreslení grafu
    plt.figure(figsize=(11, 6), dpi=100)
    plt.bar(
        x,
        aggregated_frequencies[1:],  # bez první nuly
        edgecolor=HISTOGRAM_EDGECOLOR_1,
        color=HISTOGRAM_COLOR_1,
        linewidth=1.2,
        width=1.7
    )

    # Popisky a osy
    plt.xlabel('Počet recenzí', fontsize=16)
    plt.ylabel('Počet uživatelů (log měřítko)', fontsize=16)
    plt.title('Histogram počtu uživatelských recenzí', fontsize=20)
    plt.yscale('log')

    plt.xticks(x, x_labels, fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlim(x[0] - 2.9, x[-1] + 2.9)
    plt.ylim(top=(10**5)/2.35)

    # Čísla nad sloupci
    for i, xpos in enumerate(x):
        count = aggregated_frequencies[i + 1]
        if count > 0:
            plt.text(xpos, count * 1.1, str(count), ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_OUTPUT_PATH, "histogram-of-user-reviews-all.pdf"))
    plt.show()

def plot_users_reviews_count_pie_chart(reviews_frequencies: List[int]):

    plt.figure(figsize=(10, 6), dpi=100)
    # Calculate frequencies
    max_reviews = max(reviews_frequencies)
    frequencies = [reviews_frequencies.count(i) for i in range(1, max_reviews + 1)]
    labels = [f'{i} reviews' for i in range(1, max_reviews + 1)]

    # Plot pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(frequencies, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of User Reviews')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def load_places_data():
    d_loader = DatasetLoader()
    places_data_raw = d_loader.load_data_raw()

    places_data = json_places_to_dictionary(places_data_raw)
    return places_data

def aggregate_user_reviews_google_places_API() -> Dict[str, List[object]]:
    """
        Aggregates user reviews from the original places.json file.
        Returns a dictionary mapping of user name to a list of places user voted for.
    """

    places_data = load_places_data()
    user_reviewed_places = get_users_reviewed_places_dictionary(places_data)


    return user_reviewed_places


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

def store_user_places_percentages(frequencies: Dict[str, list[str]], upper_frequency_limit: int, file_name, print_console: bool = False):
    """
    Stores the user reviews frequencies - meaning how many users have reviewed 1, 2, 3, ... places.
    To csv file.

    upper_frequency_limit: the upper limit of the number of reviews to consider as separate category.
                            Anything above this limit will be aggregated into single category.
    """


    with open(file_name, 'w') as file:
        total_percentage = 0
        for k in range(1, upper_frequency_limit + 1):
            reviewers_count = frequencies.count(k) # number of users that have reviewed k places

            if (k < upper_frequency_limit):
                percentage = round(reviewers_count / len(frequencies) * 100, 2)
                total_percentage += percentage
            else:
                percentage = round(100 - total_percentage, 2) # the rest is for the limit

            file_line = f'{k},{reviewers_count},{percentage}%\n'
            if(print_console): print(file_line)

            file.write(file_line);


def store_user_places_frequencies(frequencies: Dict[str, list[str]], upper_frequency_limit: int, file_name):
    """
    Stores the user reviews frequencies - meaning how many users

    in a json file.
    """
    with open(file_name, 'w') as file:
        json.dump(frequencies, file)

def filter_by_number_of_places(number_of_places: int, data: Dict[str, list[object]], condition_cb: Callable[[int, int], bool] = op.eq):
    """
        Filters only users that rated specific number of restaurants.
    """

    return {k: v for k, v in data.items() if condition_cb(len(v), number_of_places)}

restaurant_types = ['restaurant', 'donut_shop', 'acai_shop', 'meal_takeaway', 'bakery', 'fast_food_restaurant', 'breakfast_restaurant', 'chocolate_factory', 'hamburger_restaurant', 'brunch_restaurant', 'dog_cafe', 'japanese_restaurant', 'dessert_restaurant', 'juice_shop', 'diner', 'sushi_restaurant', 'italian_restaurant', 'pub', 'meal_delivery', 'bar_and_grill', 'chinese_restaurant', 'vegan_restaurant', 'mediterranean_restaurant', 'dessert_shop', 'wine_bar', 'american_restaurant', 'vegetarian_restaurant', 'thai_restaurant', 'chocolate_shop', 'barbecue_restaurant', 'bagel_shop', 'asian_restaurant', 'bar', 'confectionery', 'indonesian_restaurant', 'lebanese_restaurant', 'sandwich_shop', 'african_restaurant', 'spanish_restaurant', 'turkish_restaurant', 'wine_bardeli', 'vietnamese_restaurant', 'ramen_restaurant', 'steak_house', 'cafe', 'buffet_restaurant', 'fine_dining_restaurant', 'cat_cafe', 'korean_restaurant', 'deli', 'restaurant', 'tea_house', 'brazilian_restaurant', 'indian_restaurant', 'mexican_restaurant', 'pizza_restaurant', 'afghani_restaurant', 'confectioneryacai_shop', 'middle_eastern_restaurant', 'greek_restaurant', 'candy_store', 'coffee_shop', 'ice_cream_shop', 'cafeteria', 'food_court', 'french_restaurant', 'seafood_restaurant']

def categorical_coverage_analysis(aggregated_authors_places: Dict[str, List[object]]):

    aggregated_of_specific_size = filter_by_number_of_places(1, aggregated_authors_places, op.ge)

    unique_categories = {}
    sm = 0
    for key, value in list(aggregated_of_specific_size.items()): #[:10]
        places_categories = list(map(lambda i: i.get("types"), value))
        sm += len(places_categories)
        for place_cats in places_categories:
            for cat in place_cats:
                if(cat in restaurant_types):
                    if (cat not in unique_categories):
                        unique_categories[cat] = 1
                    else:
                        unique_categories[cat] += 1

    print(sm)
    print(set(restaurant_types).difference(set(unique_categories.keys())))
    print(f"{len(unique_categories)}/{len(restaurant_types)}")
    print(len(unique_categories) / len(restaurant_types))


# region Total restaurant coverage in top-n users.

def get_topn_users_coverage(aggregated_authors_places: Dict[str, List[object]], min_ratings_count: int) -> Dict[str, List[object]]:

    aggregated_of_specific_size = filter_by_number_of_places(min_ratings_count, aggregated_authors_places, op.ge)

    unique_places = {}
    for key, value in list(aggregated_of_specific_size.items()):
        for place in value:
            place_id = place.get("id")
            unique_places[place_id] = value

    return unique_places

def plot_histogram_of_results(sparsities: List[Tuple[float, pd.DataFrame]]):
    # Extract data
    sparsities_percent = [s[0] * 100 for s in sparsities]
    col_counts = [s[1].shape[1] for s in sparsities]

    # X-axis: index positions
    x = list(range(1, len(sparsities) + 1))

    # Create subplots: 2 rows, shared x-axis
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})

    # === Top subplot: Sparsity bar chart ===
    bars = axes[0].bar(x, sparsities_percent, color='skyblue', width=0.5)

    axes[0].set_xlabel('Data Point Index')
    axes[0].set_ylabel('Sparsity (%)')
    axes[0].set_title('Sparsity per Data Point')
    axes[0].set_xticks(x)
    axes[0].set_ylim(0, max(sparsities_percent) + 10)

    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    f'{height:.1f}%',
                    ha='center', va='bottom',
                    fontsize=8, rotation=45)

    # === Bottom subplot: Column count line plot ===
    axes[1].plot(x, col_counts, color='green', marker='s', linestyle='-', label='Column Count')
    axes[1].set_xlabel('Data Point Index')
    axes[1].set_ylabel('Number of Columns')
    axes[1].set_title('Column Count per Data Point')
    axes[1].set_xticks(x)
    axes[1].set_ylim(0, max(col_counts) + 5)

    for i, count in enumerate(col_counts):
        axes[1].text(x[i], count + 0.3, str(count), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

#endregion

if __name__ == "__main__":
    aggregated_authors_places = aggregate_user_reviews_google_places_API()

    reviews_frequencies = get_users_reviews_count_frequencies(aggregated_authors_places)
    plot_users_reviews_count_histogram(reviews_frequencies, 31)
