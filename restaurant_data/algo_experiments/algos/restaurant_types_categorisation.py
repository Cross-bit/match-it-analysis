RESTAURANT_TYPES_GROUPS = {
    "african": [
        "african_restaurant"
    ],
    "american": [
        "american_restaurant", "diner", "steak_house", "brunch_restaurant", "breakfast_restaurant"
    ],
    "asian": [
        "asian_restaurant", "japanese_restaurant", "sushi_restaurant", "chinese_restaurant",
        "thai_restaurant", "indonesian_restaurant", "vietnamese_restaurant", "korean_restaurant",
        "ramen_restaurant"
    ],
    "bars_alcohol": [
        "bar", "pub", "bar_and_grill", "wine_bar", "wine_bardeli"
    ],
    "buffet_dining": [
        "buffet_restaurant", "cafeteria", "food_court", "restaurant"
    ],
    "cafes_tea": [
        "cafe", "coffee_shop", "tea_house", "cat_cafe", "dog_cafe"
    ],
    "dessert_sweets": [
        "dessert_restaurant", "dessert_shop", "ice_cream_shop", "donut_shop", "chocolate_shop",
        "chocolate_factory", "candy_store", "confectionery", "bakery", "bagel_shop"
    ],
    "european": [
        "italian_restaurant", "spanish_restaurant", "french_restaurant", "greek_restaurant"
    ],
    "fast_food_takeaway": [
        "fast_food_restaurant", "hamburger_restaurant", "pizza_restaurant", "meal_takeaway",
        "meal_delivery", "sandwich_shop"
    ],
    "fine_dining": [
        "fine_dining_restaurant"
    ],
    "latin_american": [
        "brazilian_restaurant", "mexican_restaurant"
    ],
    "middle_eastern": [
        "lebanese_restaurant", "turkish_restaurant", "middle_eastern_restaurant", "afghani_restaurant"
    ],
    "other": [
        "deli"
    ],
    "south_asian": [
        "indian_restaurant"
    ],
    "vegetarian_healthy": [
        "vegan_restaurant", "vegetarian_restaurant", "juice_shop", "acai_shop", "confectioneryacai_shop"
    ]
}

def type_to_group():

    restaurant_types = {}
    for category, types in RESTAURANT_TYPES_GROUPS.items():
        for type in types:
            restaurant_types.update({type: category})

    return restaurant_types


TYPE_TO_GROUP_MAPPING = type_to_group()

