import numpy as np
from restaurant_data.algo_experiments.algos.restaurant_types_categorisation import TYPE_TO_GROUP_MAPPING, RESTAURANT_TYPES_GROUPS
from typing import List

class PlacesAPI2RepConvertor:

    # Mappings of restaurant places API types to human readable labels
    restaurant_types = [
                    'donut_shop',
                    'acai_shop',
                    'meal_takeaway',
                    'bakery',
                    'fast_food_restaurant',
                    'breakfast_restaurant',
                    'chocolate_factory',
                    'hamburger_restaurant',
                    'brunch_restaurant',
                    'dog_cafe',
                    'japanese_restaurant',
                    'dessert_restaurant',
                    'juice_shop',
                    'diner',
                    'sushi_restaurant',
                    'italian_restaurant',
                    'pub',
                    'meal_delivery',
                    'bar_and_grill',
                    'chinese_restaurant',
                    'vegan_restaurant',
                    'mediterranean_restaurant',
                    'dessert_shop',
                    'wine_bar',
                    'american_restaurant',
                    'vegetarian_restaurant',
                    'thai_restaurant',
                    'chocolate_shop',
                    'barbecue_restaurant',
                    'bagel_shop',
                    'asian_restaurant',
                    'bar',
                    'confectionery',
                    'indonesian_restaurant',
                    'lebanese_restaurant',
                    'sandwich_shop',
                    'african_restaurant',
                    'spanish_restaurant',
                    'turkish_restaurant',
                    'wine_bardeli',
                    'vietnamese_restaurant',
                    'ramen_restaurant',
                    'steak_house',
                    'cafe',
                    'buffet_restaurant',
                    'fine_dining_restaurant',
                    'cat_cafe',
                    'korean_restaurant',
                    'deli',
                    'restaurant',
                    'tea_house',
                    'brazilian_restaurant',
                    'indian_restaurant',
                    'mexican_restaurant',
                    'pizza_restaurant',
                    'afghani_restaurant',
                    'confectioneryacai_shop',
                    'middle_eastern_restaurant',
                    'greek_restaurant',
                    'candy_store',
                    'coffee_shop',
                    'ice_cream_shop',
                    'cafeteria',
                    'food_court',
                    'french_restaurant',
                    'seafood_restaurant']

    # Places api price ranges

    price_levels=["PRICE_LEVEL_UNSPECIFIED", "PRICE_LEVEL_FREE", "PRICE_LEVEL_INEXPENSIVE", "PRICE_LEVEL_MODERATE", "PRICE_LEVEL_EXPENSIVE", "PRICE_LEVEL_VERY_EXPENSIVE"]

    @staticmethod
    def price_level_to_number(price_level):
        if price_level not in PlacesAPI2RepConvertor.price_levels:
            return 0

        return PlacesAPI2RepConvertor.price_levels.index(price_level)

    @staticmethod
    def get_restaurant_type_vector(restaurant_types):

        type_vector = np.zeros(len(RESTAURANT_TYPES_GROUPS))
        group_options = list(RESTAURANT_TYPES_GROUPS.keys())

        for type in restaurant_types:
            if (type in TYPE_TO_GROUP_MAPPING):
                group_index = group_options.index(TYPE_TO_GROUP_MAPPING[type])
                type_vector[group_index] = 1
        return type_vector

    @staticmethod
    def get_vector_dim() -> int:
        return len(PlacesAPI2RepConvertor.get_representation_names())

    @staticmethod
    def get_representation_names():
        return list(RESTAURANT_TYPES_GROUPS.keys()) + ["rating", "price_level"]

    @staticmethod
    def places_api_data_to_vector(place_data) -> List[float]:

        restaurant_types = place_data.get("types", [])
        rating = float(place_data.get("rating", 0)) / 5
        price_level = float(PlacesAPI2RepConvertor.price_level_to_number(place_data.get("priceLevel"))) / len(PlacesAPI2RepConvertor.price_levels)

        type_vector = PlacesAPI2RepConvertor.get_restaurant_type_vector(restaurant_types)

        numeric_vector = np.array([rating, price_level])

        return np.concatenate([type_vector, numeric_vector]).tolist()

