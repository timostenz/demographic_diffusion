import random
import pandas as pd
import numpy as np
from aesthetic_inference import *

clicktopurchase_ratio = {
    ('appliances', 'Air Conditioners'): 0.05,
    ('appliances', 'All Appliances'): 0.05,
    ('car & motorbike', 'All Car & Motorbike Products'): 0.06,
    ('tv, audio & cameras', 'All Electronics'): 0.07,
    ('sports & fitness', 'All Exercise & Fitness'): 0.08,
    ('grocery & gourmet foods', 'All Grocery & Gourmet Foods'): 0.12,
    ('home & kitchen', 'All Home & Kitchen'): 0.10,
    ('pet supplies', 'All Pet Supplies'): 0.09,
    ('sports & fitness', 'All Sports, Fitness & Outdoors'): 0.08,
    ('stores', 'Amazon Fashion'): 0.10,
    ('toys & baby products', 'Baby Bath, Skin & Grooming'): 0.12,
    ('kids\' fashion', 'Baby Fashion'): 0.11,
    ('toys & baby products', 'Baby Products'): 0.11,
    ('bags & luggage', 'Backpacks'): 0.09,
    ('sports & fitness', 'Badminton'): 0.07,
    ('accessories', 'Bags & Luggage'): 0.08,
    ('women\'s shoes', 'Ballerinas'): 0.10,
    ('beauty & health', 'Beauty & Grooming'): 0.11,
    ('home & kitchen', 'Bedroom Linen'): 0.10,
    ('tv, audio & cameras', 'Camera Accessories'): 0.07,
    ('tv, audio & cameras', 'Cameras'): 0.06,
    ('sports & fitness', 'Camping & Hiking'): 0.08,
    ('car & motorbike', 'Car & Bike Care'): 0.07,
    ('car & motorbike', 'Car Accessories'): 0.07,
    ('car & motorbike', 'Car Electronics'): 0.07,
    ('car & motorbike', 'Car Parts'): 0.07,
    ('sports & fitness', 'Cardio Equipment'): 0.08,
    ('men\'s shoes', 'Casual Shoes'): 0.10,
    ('women\'s clothing', 'Clothing'): 0.11,
    ('grocery & gourmet foods', 'Coffee, Tea & Beverages'): 0.13,
    ('sports & fitness', 'Cricket'): 0.08,
    ('sports & fitness', 'Cycling'): 0.07,
    ('toys & baby products', 'Diapers'): 0.12,
    ('beauty & health', 'Diet & Nutrition'): 0.11,
    ('pet supplies', 'Dog supplies'): 0.09,
    ('women\'s clothing', 'Ethnic Wear'): 0.11,
    ('accessories', 'Fashion & Silver Jewellery'): 0.10,
    ('stores', 'Fashion Sales & Deals'): 0.12,
    ('women\'s shoes', 'Fashion Sandals'): 0.10,
    ('sports & fitness', 'Fitness Accessories'): 0.08,
    ('sports & fitness', 'Football'): 0.08,
    ('men\'s shoes', 'Formal Shoes'): 0.10,
    ('home & kitchen', 'Furniture'): 0.08,
    ('home & kitchen', 'Garden & Outdoors'): 0.09,
    ('accessories', 'Gold & Diamond Jewellery'): 0.10,
    ('accessories', 'Handbags & Clutches'): 0.09,
    ('tv, audio & cameras', 'Headphones'): 0.07,
    ('beauty & health', 'Health & Personal Care'): 0.10,
    ('appliances', 'Heating & Cooling Appliances'): 0.05,
    ('tv, audio & cameras', 'Home Audio & Theater'): 0.06,
    ('home & kitchen', 'Home Décor'): 0.10,
    ('tv, audio & cameras', 'Home Entertainment Systems'): 0.06,
    ('home & kitchen', 'Home Furnishing'): 0.10,
    ('home & kitchen', 'Home Improvement'): 0.09,
    ('home & kitchen', 'Home Storage'): 0.09,
    ('beauty & health', 'Household Supplies'): 0.12,
    ('home & kitchen', 'Indoor Lighting'): 0.09,
    ('industrial supplies', 'Industrial & Scientific Supplies'): 0.05,
    ('men\'s clothing', 'Innerwear'): 0.10,
    ('toys & baby products', 'International Toy Store'): 0.12,
    ('industrial supplies', 'Janitorial & Sanitation Supplies'): 0.05,
    ('men\'s clothing', 'Jeans'): 0.10,
    ('accessories', 'Jewellery'): 0.10,
    ('kids\' fashion', 'Kids\' Clothing'): 0.11,
    ('kids\' fashion', 'Kids\' Fashion'): 0.11,
    ('kids\' fashion', 'Kids\' Shoes'): 0.11,
    ('kids\' fashion', 'Kids\' Watches'): 0.11,
    ('home & kitchen', 'Kitchen & Dining'): 0.10,
    ('appliances', 'Kitchen & Home Appliances'): 0.08,
    ('home & kitchen', 'Kitchen Storage & Containers'): 0.09,
    ('industrial supplies', 'Lab & Scientific'): 0.05,
    ('women\'s clothing', 'Lingerie & Nightwear'): 0.10,
    ('beauty & health', 'Luxury Beauty'): 0.11,
    ('beauty & health', 'Make-up'): 0.12,
    ('stores', 'Men\'s Fashion'): 0.11,
    ('car & motorbike', 'Motorbike Accessories & Parts'): 0.06,
    ('music', 'Musical Instruments & Professional Audio'): 0.06,
    ('toys & baby products', 'Nursing & Feeding'): 0.11,
    ('beauty & health', 'Personal Care Appliances'): 0.09,
    ('appliances', 'Refrigerators'): 0.06,
    ('home, kitchen, pets', 'Refurbished & Open Box'): 0.05,
    ('bags & luggage', 'Rucksacks'): 0.08,
    ('sports & fitness', 'Running'): 0.08,
    ('kids\' fashion', 'School Bags'): 0.11,
    ('tv, audio & cameras', 'Security Cameras'): 0.07,
    ('home & kitchen', 'Sewing & Craft Supplies'): 0.09,
    ('men\'s clothing', 'Shirts'): 0.10,
    ('women\'s shoes', 'Shoes'): 0.10,
    ('grocery & gourmet foods', 'Snack Foods'): 0.12,
    ('tv, audio & cameras', 'Speakers'): 0.06,
    ('men\'s shoes', 'Sports Shoes'): 0.10,
    ('stores', 'Sportswear'): 0.10,
    ('toys & baby products', 'STEM Toys Store'): 0.11,
    ('sports & fitness', 'Strength Training'): 0.08,
    ('toys & baby products', 'Strollers & Prams'): 0.12,
    ('bags & luggage', 'Suitcases & Trolley Bags'): 0.08,
    ('accessories', 'Sunglasses'): 0.10,
    ('men\'s clothing', 'T-shirts & Polos'): 0.10,
    ('tv, audio & cameras', 'Televisions'): 0.06,
    ('industrial supplies', 'Test, Measure & Inspect'): 0.05,
    ('stores', 'The Designer Boutique'): 0.12,
    ('toys & baby products', 'Toys & Games'): 0.11,
    ('toys & baby products', 'Toys Gifting Store'): 0.11,
    ('bags & luggage', 'Travel Accessories'): 0.08,
    ('bags & luggage', 'Travel Duffles'): 0.08,
    ('beauty & health', 'Value Bazaar'): 0.12,
    ('bags & luggage', 'Wallets'): 0.09,
    ('appliances', 'Washing Machines'): 0.05,
    ('accessories', 'Watches'): 0.10,
    ('women\'s clothing', 'Western Wear'): 0.11,
    ('stores', 'Women\'s Fashion'): 0.11,
    ('sports & fitness', 'Yoga'): 0.08,
}
base_ctrs = {
    ('appliances', 'Air Conditioners'): 0.0035,
    ('appliances', 'All Appliances'): 0.004,
    ('car & motorbike', 'All Car & Motorbike Products'): 0.003,
    ('tv, audio & cameras', 'All Electronics'): 0.004,
    ('sports & fitness', 'All Exercise & Fitness'): 0.005,
    ('grocery & gourmet foods', 'All Grocery & Gourmet Foods'): 0.006,
    ('home & kitchen', 'All Home & Kitchen'): 0.0045,
    ('pet supplies', 'All Pet Supplies'): 0.0045,
    ('sports & fitness', 'All Sports, Fitness & Outdoors'): 0.005,
    ('stores', 'Amazon Fashion'): 0.0065,
    ('toys & baby products', 'Baby Bath, Skin & Grooming'): 0.007,
    ('kids\' fashion', 'Baby Fashion'): 0.0075,
    ('toys & baby products', 'Baby Products'): 0.0075,
    ('bags & luggage', 'Backpacks'): 0.005,
    ('sports & fitness', 'Badminton'): 0.0045,
    ('accessories', 'Bags & Luggage'): 0.005,
    ('women\'s shoes', 'Ballerinas'): 0.006,
    ('beauty & health', 'Beauty & Grooming'): 0.0055,
    ('home & kitchen', 'Bedroom Linen'): 0.004,
    ('tv, audio & cameras', 'Camera Accessories'): 0.0035,
    ('tv, audio & cameras', 'Cameras'): 0.0035,
    ('sports & fitness', 'Camping & Hiking'): 0.005,
    ('car & motorbike', 'Car & Bike Care'): 0.003,
    ('car & motorbike', 'Car Accessories'): 0.003,
    ('car & motorbike', 'Car Electronics'): 0.003,
    ('car & motorbike', 'Car Parts'): 0.003,
    ('sports & fitness', 'Cardio Equipment'): 0.0045,
    ('men\'s shoes', 'Casual Shoes'): 0.0065,
    ('women\'s clothing', 'Clothing'): 0.006,
    ('grocery & gourmet foods', 'Coffee, Tea & Beverages'): 0.0055,
    ('sports & fitness', 'Cricket'): 0.0045,
    ('sports & fitness', 'Cycling'): 0.0045,
    ('toys & baby products', 'Diapers'): 0.0075,
    ('beauty & health', 'Diet & Nutrition'): 0.005,
    ('pet supplies', 'Dog supplies'): 0.0045,
    ('women\'s clothing', 'Ethnic Wear'): 0.0065,
    ('accessories', 'Fashion & Silver Jewellery'): 0.007,
    ('stores', 'Fashion Sales & Deals'): 0.007,
    ('women\'s shoes', 'Fashion Sandals'): 0.0065,
    ('sports & fitness', 'Fitness Accessories'): 0.005,
    ('sports & fitness', 'Football'): 0.0045,
    ('men\'s shoes', 'Formal Shoes'): 0.0065,
    ('home & kitchen', 'Furniture'): 0.004,
    ('home & kitchen', 'Garden & Outdoors'): 0.004,
    ('accessories', 'Gold & Diamond Jewellery'): 0.007,
    ('accessories', 'Handbags & Clutches'): 0.006,
    ('tv, audio & cameras', 'Headphones'): 0.0035,
    ('beauty & health', 'Health & Personal Care'): 0.005,
    ('appliances', 'Heating & Cooling Appliances'): 0.003,
    ('tv, audio & cameras', 'Home Audio & Theater'): 0.0035,
    ('home & kitchen', 'Home Décor'): 0.004,
    ('tv, audio & cameras', 'Home Entertainment Systems'): 0.0035,
    ('home & kitchen', 'Home Furnishing'): 0.004,
    ('home & kitchen', 'Home Improvement'): 0.004,
    ('home & kitchen', 'Home Storage'): 0.004,
    ('beauty & health', 'Household Supplies'): 0.005,
    ('home & kitchen', 'Indoor Lighting'): 0.004,
    ('industrial supplies', 'Industrial & Scientific Supplies'): 0.003,
    ('men\'s clothing', 'Innerwear'): 0.0065,
    ('toys & baby products', 'International Toy Store'): 0.007,
    ('industrial supplies', 'Janitorial & Sanitation Supplies'): 0.003,
    ('men\'s clothing', 'Jeans'): 0.0065,
    ('accessories', 'Jewellery'): 0.007,
    ('kids\' fashion', 'Kids\' Clothing'): 0.007,
    ('kids\' fashion', 'Kids\' Fashion'): 0.007,
    ('kids\' fashion', 'Kids\' Shoes'): 0.007,
    ('kids\' fashion', 'Kids\' Watches'): 0.007,
    ('home & kitchen', 'Kitchen & Dining'): 0.004,
    ('appliances', 'Kitchen & Home Appliances'): 0.004,
    ('home & kitchen', 'Kitchen Storage & Containers'): 0.004,
    ('industrial supplies', 'Lab & Scientific'): 0.003,
    ('women\'s clothing', 'Lingerie & Nightwear'): 0.0065,
    ('beauty & health', 'Luxury Beauty'): 0.0055,
    ('beauty & health', 'Make-up'): 0.0055,
    ('stores', 'Men\'s Fashion'): 0.0065,
    ('car & motorbike', 'Motorbike Accessories & Parts'): 0.003,
    ('music', 'Musical Instruments & Professional Audio'): 0.003,
    ('toys & baby products', 'Nursing & Feeding'): 0.0075,
    ('beauty & health', 'Personal Care Appliances'): 0.0055,
    ('appliances', 'Refrigerators'): 0.0035,
    ('home, kitchen, pets', 'Refurbished & Open Box'): 0.003,
    ('bags & luggage', 'Rucksacks'): 0.005,
    ('sports & fitness', 'Running'): 0.0045,
    ('kids\' fashion', 'School Bags'): 0.007,
    ('tv, audio & cameras', 'Security Cameras'): 0.0035,
    ('home & kitchen', 'Sewing & Craft Supplies'): 0.004,
    ('men\'s clothing', 'Shirts'): 0.0065,
    ('women\'s shoes', 'Shoes'): 0.0065,
    ('grocery & gourmet foods', 'Snack Foods'): 0.0055,
    ('tv, audio & cameras', 'Speakers'): 0.0035,
    ('men\'s shoes', 'Sports Shoes'): 0.0065,
    ('stores', 'Sportswear'): 0.0065,
    ('toys & baby products', 'STEM Toys Store'): 0.007,
    ('sports & fitness', 'Strength Training'): 0.005,
    ('toys & baby products', 'Strollers & Prams'): 0.0075,
    ('bags & luggage', 'Suitcases & Trolley Bags'): 0.005,
    ('accessories', 'Sunglasses'): 0.007,
    ('men\'s clothing', 'T-shirts & Polos'): 0.0065,
    ('tv, audio & cameras', 'Televisions'): 0.0035,
    ('industrial supplies', 'Test, Measure & Inspect'): 0.003,
    ('stores', 'The Designer Boutique'): 0.007,
    ('toys & baby products', 'Toys & Games'): 0.007,
    ('toys & baby products', 'Toys Gifting Store'): 0.007,
    ('bags & luggage', 'Travel Accessories'): 0.005,
    ('bags & luggage', 'Travel Duffles'): 0.005,
    ('beauty & health', 'Value Bazaar'): 0.005,
    ('bags & luggage', 'Wallets'): 0.005,
    ('appliances', 'Washing Machines'): 0.0035,
    ('accessories', 'Watches'): 0.007,
    ('women\'s clothing', 'Western Wear'): 0.0065,
    ('stores', 'Women\'s Fashion'): 0.0065,
    ('sports & fitness', 'Yoga'): 0.005,
}
heuristics = {
    ('appliances', 'Air Conditioners'): {
        'male_probability': 0.6,
        'age_distribution': [0.10, 0.25, 0.25, 0.20, 0.10, 0.07, 0.03],
    },
    ('appliances', 'All Appliances'): {
        'male_probability': 0.65,
        'age_distribution': [0.08, 0.22, 0.25, 0.25, 0.12, 0.06, 0.02],
    },
    ('car & motorbike', 'All Car & Motorbike Products'): {
        'male_probability': 0.7,
        'age_distribution': [0.15, 0.30, 0.25, 0.15, 0.10, 0.03, 0.02],
    },
    ('tv, audio & cameras', 'All Electronics'): {
        'male_probability': 0.65,
        'age_distribution': [0.12, 0.28, 0.30, 0.18, 0.07, 0.03, 0.02],
    },
    ('sports & fitness', 'All Exercise & Fitness'): {
        'male_probability': 0.6,
        'age_distribution': [0.18, 0.30, 0.25, 0.15, 0.07, 0.03, 0.02],
    },
    ('grocery & gourmet foods', 'All Grocery & Gourmet Foods'): {
        'male_probability': 0.4,
        'age_distribution': [0.10, 0.25, 0.30, 0.20, 0.10, 0.03, 0.02],
    },
    ('home & kitchen', 'All Home & Kitchen'): {
        'male_probability': 0.5,
        'age_distribution': [0.12, 0.25, 0.28, 0.20, 0.10, 0.03, 0.02],
    },
    ('pet supplies', 'All Pet Supplies'): {
        'male_probability': 0.45,
        'age_distribution': [0.08, 0.22, 0.30, 0.25, 0.10, 0.03, 0.02],
    },
    ('sports & fitness', 'All Sports, Fitness & Outdoors'): {
        'male_probability': 0.65,
        'age_distribution': [0.20, 0.35, 0.25, 0.12, 0.05, 0.02, 0.01],
    },
    ('stores', 'Amazon Fashion'): {
        'male_probability': 0.4,
        'age_distribution': [0.18, 0.35, 0.25, 0.12, 0.05, 0.03, 0.02],
    },
    ('toys & baby products', 'Baby Bath, Skin & Grooming'): {
        'male_probability': 0.3,
        'age_distribution': [0.05, 0.20, 0.30, 0.25, 0.15, 0.03, 0.02],
    },
    ('kids\' fashion', 'Baby Fashion'): {
        'male_probability': 0.35,
        'age_distribution': [0.10, 0.30, 0.35, 0.15, 0.07, 0.02, 0.01],
    },
    ('toys & baby products', 'Baby Products'): {
        'male_probability': 0.4,
        'age_distribution': [0.05, 0.18, 0.35, 0.30, 0.10, 0.02, 0.01],
    },
    ('bags & luggage', 'Backpacks'): {
        'male_probability': 0.55,
        'age_distribution': [0.20, 0.35, 0.30, 0.10, 0.03, 0.01, 0.01],
    },
    ('sports & fitness', 'Badminton'): {
        'male_probability': 0.5,
        'age_distribution': [0.15, 0.40, 0.25, 0.10, 0.05, 0.03, 0.02],
    },
    ('accessories', 'Bags & Luggage'): {
        'male_probability': 0.5,
        'age_distribution': [0.15, 0.30, 0.30, 0.15, 0.07, 0.02, 0.01],
    },
    ('women\'s shoes', 'Ballerinas'): {
        'male_probability': 0.05,
        'age_distribution': [0.15, 0.40, 0.25, 0.10, 0.05, 0.03, 0.02],
    },
    ('beauty & health', 'Beauty & Grooming'): {
        'male_probability': 0.3,
        'age_distribution': [0.12, 0.35, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('home & kitchen', 'Bedroom Linen'): {
        'male_probability': 0.4,
        'age_distribution': [0.10, 0.25, 0.30, 0.25, 0.07, 0.02, 0.01],
    },
    ('tv, audio & cameras', 'Camera Accessories'): {
        'male_probability': 0.7,
        'age_distribution': [0.15, 0.35, 0.30, 0.12, 0.05, 0.02, 0.01],
    },
    ('tv, audio & cameras', 'Cameras'): {
        'male_probability': 0.75,
        'age_distribution': [0.12, 0.30, 0.35, 0.15, 0.05, 0.02, 0.01],
    },
    ('sports & fitness', 'Camping & Hiking'): {
        'male_probability': 0.75,
        'age_distribution': [0.18, 0.35, 0.30, 0.12, 0.03, 0.01, 0.01],
    },
    ('car & motorbike', 'Car & Bike Care'): {
        'male_probability': 0.75,
        'age_distribution': [0.20, 0.30, 0.25, 0.15, 0.05, 0.03, 0.02],
    },
    ('car & motorbike', 'Car Accessories'): {
        'male_probability': 0.7,
        'age_distribution': [0.15, 0.30, 0.30, 0.15, 0.07, 0.02, 0.01],
    },
        ('car & motorbike', 'Car Electronics'): {
        'male_probability': 0.85,
        'age_distribution': [0.15, 0.32, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('car & motorbike', 'Car Parts'): {
        'male_probability': 0.88,
        'age_distribution': [0.18, 0.30, 0.28, 0.15, 0.05, 0.03, 0.01],
    },
    ('sports & fitness', 'Cardio Equipment'): {
        'male_probability': 0.5,
        'age_distribution': [0.20, 0.35, 0.25, 0.12, 0.05, 0.02, 0.01],
    },
    ('men\'s shoes', 'Casual Shoes'): {
        'male_probability': 0.9,
        'age_distribution': [0.18, 0.40, 0.30, 0.08, 0.02, 0.01, 0.01],
    },
    ('women\'s clothing', 'Clothing'): {
        'male_probability': 0.15,
        'age_distribution': [0.10, 0.40, 0.30, 0.10, 0.05, 0.03, 0.02],
    },
    ('grocery & gourmet foods', 'Coffee, Tea & Beverages'): {
        'male_probability': 0.4,
        'age_distribution': [0.15, 0.30, 0.25, 0.20, 0.07, 0.02, 0.01],
    },
    ('sports & fitness', 'Cricket'): {
        'male_probability': 0.7,
        'age_distribution': [0.25, 0.35, 0.25, 0.10, 0.03, 0.01, 0.01],
    },
    ('sports & fitness', 'Cycling'): {
        'male_probability': 0.65,
        'age_distribution': [0.20, 0.35, 0.25, 0.10, 0.07, 0.02, 0.01],
    },
    ('toys & baby products', 'Diapers'): {
        'male_probability': 0.35,
        'age_distribution': [0.05, 0.20, 0.35, 0.25, 0.10, 0.03, 0.02],
    },
    ('beauty & health', 'Diet & Nutrition'): {
        'male_probability': 0.5,
        'age_distribution': [0.12, 0.30, 0.30, 0.15, 0.08, 0.03, 0.02],
    },
    ('pet supplies', 'Dog Supplies'): {
        'male_probability': 0.5,
        'age_distribution': [0.08, 0.25, 0.30, 0.25, 0.07, 0.03, 0.02],
    },
    ('women\'s clothing', 'Ethnic Wear'): {
        'male_probability': 0.2,
        'age_distribution': [0.08, 0.35, 0.30, 0.15, 0.07, 0.03, 0.02],
    },
    ('accessories', 'Fashion & Silver Jewellery'): {
        'male_probability': 0.3,
        'age_distribution': [0.10, 0.30, 0.30, 0.20, 0.07, 0.02, 0.01],
    },
    ('stores', 'Fashion Sales & Deals'): {
        'male_probability': 0.4,
        'age_distribution': [0.15, 0.35, 0.30, 0.15, 0.03, 0.01, 0.01],
    },
    ('women\'s shoes', 'Fashion Sandals'): {
        'male_probability': 0.05,
        'age_distribution': [0.10, 0.40, 0.30, 0.15, 0.03, 0.01, 0.01],
    },
    ('sports & fitness', 'Fitness Accessories'): {
        'male_probability': 0.6,
        'age_distribution': [0.20, 0.35, 0.25, 0.10, 0.05, 0.03, 0.02],
    },
    ('sports & fitness', 'Football'): {
        'male_probability': 0.8,
        'age_distribution': [0.25, 0.40, 0.25, 0.05, 0.03, 0.01, 0.01],
    },
    ('men\'s shoes', 'Formal Shoes'): {
        'male_probability': 0.95,
        'age_distribution': [0.15, 0.35, 0.30, 0.15, 0.03, 0.01, 0.01],
    },
        ('home & kitchen', 'Furniture'): {
        'male_probability': 0.28,
        'age_distribution': [0.10, 0.25, 0.30, 0.20, 0.10, 0.03, 0.02],
    },
    ('home & kitchen', 'Garden & Outdoors'): {
        'male_probability': 0.39,
        'age_distribution': [0.12, 0.28, 0.30, 0.18, 0.07, 0.03, 0.02],
    },
    ('accessories', 'Gold & Diamond Jewellery'): {
        'male_probability': 0.30,
        'age_distribution': [0.05, 0.20, 0.30, 0.25, 0.15, 0.03, 0.02],
    },
    ('accessories', 'Handbags & Clutches'): {
        'male_probability': 0.20,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('tv, audio & cameras', 'Headphones'): {
        'male_probability': 0.50,
        'age_distribution': [0.25, 0.30, 0.25, 0.12, 0.05, 0.02, 0.01],
    },
    ('beauty & health', 'Health & Personal Care'): {
        'male_probability': 0.25,
        'age_distribution': [0.10, 0.25, 0.30, 0.20, 0.10, 0.03, 0.02],
    },
    ('appliances', 'Heating & Cooling Appliances'): {
        'male_probability': 0.50,
        'age_distribution': [0.08, 0.22, 0.30, 0.25, 0.10, 0.03, 0.02],
    },
    ('tv, audio & cameras', 'Home Audio & Theater'): {
        'male_probability': 0.55,
        'age_distribution': [0.12, 0.28, 0.30, 0.18, 0.07, 0.03, 0.02],
    },
    ('home & kitchen', 'Home Décor'): {
        'male_probability': 0.20,
        'age_distribution': [0.10, 0.25, 0.30, 0.20, 0.10, 0.03, 0.02],
    },
    ('tv, audio & cameras', 'Home Entertainment Systems'): {
        'male_probability': 0.50,
        'age_distribution': [0.12, 0.30, 0.30, 0.18, 0.07, 0.02, 0.01],
    },
    ('home & kitchen', 'Home Furnishing'): {
        'male_probability': 0.30,
        'age_distribution': [0.08, 0.22, 0.30, 0.25, 0.10, 0.03, 0.02],
    },
    ('home & kitchen', 'Home Improvement'): {
        'male_probability': 0.29,
        'age_distribution': [0.12, 0.25, 0.28, 0.20, 0.10, 0.03, 0.02],
    },
    ('home & kitchen', 'Home Storage'): {
        'male_probability': 0.30,
        'age_distribution': [0.10, 0.25, 0.30, 0.20, 0.10, 0.03, 0.02],
    },
    ('beauty & health', 'Household Supplies'): {
        'male_probability': 0.22,
        'age_distribution': [0.10, 0.30, 0.30, 0.20, 0.07, 0.02, 0.01],
    },
    ('home & kitchen', 'Indoor Lighting'): {
        'male_probability': 0.50,
        'age_distribution': [0.12, 0.28, 0.30, 0.18, 0.07, 0.03, 0.02],
    },
    ('industrial supplies', 'Industrial & Scientific Supplies'): {
        'male_probability': 0.5,
        'age_distribution': [0.15, 0.30, 0.30, 0.15, 0.07, 0.02, 0.01],
    },
    ('men\'s clothing', 'Innerwear'): {
        'male_probability': 0.70,
        'age_distribution': [0.12, 0.35, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('toys & baby products', 'International Toy Store'): {
        'male_probability': 0.32,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('industrial supplies', 'Janitorial & Sanitation Supplies'): {
        'male_probability': 0.5,
        'age_distribution': [0.15, 0.30, 0.30, 0.15, 0.07, 0.02, 0.01],
    },
    ('men\'s clothing', 'Jeans'): {
        'male_probability': 0.8,
        'age_distribution': [0.12, 0.40, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('accessories', 'Jewellery'): {
        'male_probability': 0.40,
        'age_distribution': [0.05, 0.20, 0.30, 0.25, 0.15, 0.03, 0.02],
    },
    ('kids\' fashion', 'Kids\' Clothing'): {
        'male_probability': 0.28,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('kids\' fashion', 'Kids\' Fashion'): {
        'male_probability': 0.24,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('kids\' fashion', 'Kids\' Shoes'): {
        'male_probability': 0.25,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('kids\' fashion', 'Kids\' Watches'): {
        'male_probability': 0.26,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
        ('home & kitchen', 'Kitchen & Dining'): {
        'male_probability': 0.30,
        'age_distribution': [0.10, 0.28, 0.30, 0.20, 0.08, 0.03, 0.01],
    },
    ('appliances', 'Kitchen & Home Appliances'): {
        'male_probability': 0.34,
        'age_distribution': [0.12, 0.30, 0.28, 0.18, 0.08, 0.03, 0.01],
    },
    ('home & kitchen', 'Kitchen Storage & Containers'): {
        'male_probability': 0.35,
        'age_distribution': [0.10, 0.25, 0.30, 0.20, 0.10, 0.03, 0.02],
    },
    ('industrial supplies', 'Lab & Scientific'): {
        'male_probability': 0.5,
        'age_distribution': [0.15, 0.30, 0.30, 0.15, 0.07, 0.02, 0.01],
    },
    ('women\'s clothing', 'Lingerie & Nightwear'): {
        'male_probability': 0.10,
        'age_distribution': [0.05, 0.20, 0.35, 0.25, 0.10, 0.03, 0.02],
    },
    ('beauty & health', 'Luxury Beauty'): {
        'male_probability': 0.21,
        'age_distribution': [0.05, 0.25, 0.35, 0.25, 0.08, 0.02, 0.01],
    },
    ('beauty & health', 'Make-up'): {
        'male_probability': 0.12,
        'age_distribution': [0.05, 0.25, 0.40, 0.20, 0.07, 0.02, 0.01],
    },
    ('stores', 'Men\'s Fashion'): {
        'male_probability': 0.81,
        'age_distribution': [0.15, 0.35, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('car & motorbike', 'Motorbike Accessories & Parts'): {
        'male_probability': 0.7,
        'age_distribution': [0.20, 0.40, 0.25, 0.10, 0.03, 0.01, 0.01],
    },
    ('music', 'Musical Instruments & Professional Audio'): {
        'male_probability': 0.55,
        'age_distribution': [0.10, 0.30, 0.35, 0.15, 0.07, 0.02, 0.01],
    },
    ('toys & baby products', 'Nursing & Feeding'): {
        'male_probability': 0.21,
        'age_distribution': [0.08, 0.25, 0.35, 0.25, 0.05, 0.02, 0.01],
    },
    ('beauty & health', 'Personal Care Appliances'): {
        'male_probability': 0.19,
        'age_distribution': [0.12, 0.30, 0.30, 0.20, 0.05, 0.02, 0.01],
    },
    ('appliances', 'Refrigerators'): {
        'male_probability': 0.45,
        'age_distribution': [0.12, 0.30, 0.28, 0.20, 0.08, 0.02, 0.01],
    },
    ('home, kitchen, pets', 'Refurbished & Open Box'): {
        'male_probability': 0.25,
        'age_distribution': [0.12, 0.30, 0.30, 0.18, 0.07, 0.02, 0.01],
    },
    ('bags & luggage', 'Rucksacks'): {
        'male_probability': 0.5,
        'age_distribution': [0.10, 0.28, 0.30, 0.20, 0.08, 0.03, 0.01],
    },
    ('sports & fitness', 'Running'): {
        'male_probability': 0.55,
        'age_distribution': [0.12, 0.35, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('kids\' fashion', 'School Bags'): {
        'male_probability': 0.31,
        'age_distribution': [0.10, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('tv, audio & cameras', 'Security Cameras'): {
        'male_probability': 0.6,
        'age_distribution': [0.15, 0.35, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('home & kitchen', 'Sewing & Craft Supplies'): {
        'male_probability': 0.3,
        'age_distribution': [0.10, 0.25, 0.30, 0.25, 0.08, 0.02, 0.01],
    },
    ('men\'s clothing', 'Shirts'): {
        'male_probability': 0.80,
        'age_distribution': [0.15, 0.40, 0.30, 0.10, 0.03, 0.01, 0.01],
    },
    ('women\'s shoes', 'Shoes'): {
        'male_probability': 0.10,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('grocery & gourmet foods', 'Snack Foods'): {
        'male_probability': 0.42,
        'age_distribution': [0.08, 0.25, 0.30, 0.25, 0.10, 0.02, 0.01],
    },
        ('tv, audio & cameras', 'Speakers'): {
        'male_probability': 0.56,
        'age_distribution': [0.12, 0.35, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('men\'s shoes', 'Sports Shoes'): {
        'male_probability': 0.80,
        'age_distribution': [0.15, 0.35, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('stores', 'Sportswear'): {
        'male_probability': 0.53,
        'age_distribution': [0.12, 0.30, 0.35, 0.15, 0.05, 0.02, 0.01],
    },
    ('toys & baby products', 'STEM Toys Store'): {
        'male_probability': 0.41,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('sports & fitness', 'Strength Training'): {
        'male_probability': 0.65,
        'age_distribution': [0.17, 0.3, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('toys & baby products', 'Strollers & Prams'): {
        'male_probability': 0.32,
        'age_distribution': [0.08, 0.25, 0.35, 0.25, 0.05, 0.02, 0.01],
    },
    ('bags & luggage', 'Suitcases & Trolley Bags'): {
        'male_probability': 0.38,
        'age_distribution': [0.12, 0.30, 0.35, 0.15, 0.05, 0.02, 0.01],
    },
    ('accessories', 'Sunglasses'): {
        'male_probability': 0.40,
        'age_distribution': [0.15, 0.35, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('men\'s clothing', 'T-shirts & Polos'): {
        'male_probability': 0.80,
        'age_distribution': [0.15, 0.40, 0.30, 0.10, 0.03, 0.01, 0.01],
    },
    ('tv, audio & cameras', 'Televisions'): {
        'male_probability': 0.56,
        'age_distribution': [0.12, 0.35, 0.30, 0.15, 0.05, 0.02, 0.01],
    },
    ('industrial supplies', 'Test, Measure & Inspect'): {
        'male_probability': 0.5,
        'age_distribution': [0.15, 0.30, 0.30, 0.15, 0.07, 0.02, 0.01],
    },
    ('stores', 'The Designer Boutique'): {
        'male_probability': 0.32,
        'age_distribution': [0.05, 0.20, 0.35, 0.25, 0.10, 0.03, 0.02],
    },
    ('toys & baby products', 'Toys & Games'): {
        'male_probability': 0.31,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('toys & baby products', 'Toys Gifting Store'): {
        'male_probability': 0.35,
        'age_distribution': [0.08, 0.30, 0.35, 0.20, 0.05, 0.02, 0.01],
    },
    ('bags & luggage', 'Travel Accessories'): {
        'male_probability': 0.45,
        'age_distribution': [0.12, 0.30, 0.35, 0.15, 0.05, 0.02, 0.01],
    },
    ('bags & luggage', 'Travel Duffles'): {
        'male_probability': 0.48,
        'age_distribution': [0.12, 0.30, 0.35, 0.15, 0.05, 0.02, 0.01],
    },
    ('beauty & health', 'Value Bazaar'): {
        'male_probability': 0.3,
        'age_distribution': [0.08, 0.25, 0.35, 0.25, 0.05, 0.02, 0.01],
    },
    ('bags & luggage', 'Wallets'): {
        'male_probability': 0.35,
        'age_distribution': [0.12, 0.3, 0.30, 0.15, 0.10, 0.02, 0.01],
    },
    ('appliances', 'Washing Machines'): {
        'male_probability': 0.5,
        'age_distribution': [0.12, 0.30, 0.30, 0.20, 0.08, 0.02, 0.01],
    },
    ('accessories', 'Watches'): {
        'male_probability': 0.40,
        'age_distribution': [0.15, 0.25, 0.25, 0.15, 0.15, 0.07, 0.01],
    },
    ('women\'s clothing', 'Western Wear'): {
        'male_probability': 0.1,
        'age_distribution': [0.15, 0.25, 0.25, 0.25, 0.08, 0.02, 0.01],
    },
    ('stores', 'Women\'s Fashion'): {
        'male_probability': 0.11,
        'age_distribution': [0.05, 0.25, 0.3, 0.25, 0.13, 0.02, 0.01],
    },
    ('sports & fitness', 'Yoga'): {
        'male_probability': 0.3,
        'age_distribution': [0.12, 0.30, 0.35, 0.15, 0.05, 0.02, 0.01],
    }
}

def clean_ratings(rating):
    # Ensure the input is a string, replace commas, and check if it's numeric
    if isinstance(rating, str):  # Check if the value is a string
        rating_cleaned = rating.replace(',', '')  # Remove commas
        if rating_cleaned.isdigit():  # If it is numeric after removing commas
            return int(rating_cleaned)  # Convert to float
    return np.nan  # Return NaN for non-numeric values
def estimate_purchases(data, ratings_col='corrected_no_of_ratings'):
    """
    Estimate total purchases by separating NaN and non-NaN cases.
    
    Returns:
        pd.Series: A Series containing the estimated purchases.
    """
    # Copy the data to avoid modifying the original DataFrame
    data = data.copy()

    # Step 1: For rows where ratings are not NaN
    non_nan_mask = data[ratings_col].notna()
    data.loc[non_nan_mask, 'estimated_purchases'] = (
        data.loc[non_nan_mask, ratings_col] / np.random.normal(0.02, 0.004, size=non_nan_mask.sum()) # 2% of sales leave reviews -> between 1.6 and 2.4 with uncertainty
    )

    # Step 2: For rows where ratings are NaN
    nan_mask = data[ratings_col].isna()
    data.loc[nan_mask, 'estimated_purchases'] = np.random.normal(50, 30, size=nan_mask.sum())

    # Ensure all values are integers and non-negative
    data['estimated_purchases'] = data['estimated_purchases'].apply(lambda x: max(int(x), 0))

    return data['estimated_purchases']
def convert_to_float_with_nan(series):
    """
    Convert a pandas Series to float, replacing non-convertible values with NaN,
    and leaving existing NaN values unchanged.

    Args:
        series (pd.Series): The input Series to convert.

    Returns:
        pd.Series: The Series converted to float, with invalid values as NaN.
    """
    def safe_convert(value):
        if pd.isna(value):  # If the value is already NaN, return it as is
            return np.nan
        try:
            return float(value)  # Attempt to convert to float
        except ValueError:
            return np.nan  # Replace non-convertible values with NaN
    
    return series.apply(safe_convert)
def get_ctp_rate(row):
    category_tuple = (row['main_category'], row['sub_category'])
    return clicktopurchase_ratio.get(category_tuple, None)  # Return None if no match found
def calculate_ctr(data, base_ctrs):
    """
    Calculate the click-through rate (CTR) based on category, rating, and discount.
    
    Parameters:
        main_category (str): The main category of the product.
        sub_category (str): The sub-category of the product.
        rating (float): The product's average rating (1 to 5).
        discount (float): The discount on the product as a percentage (0 to 100).
        
    Returns:
        float: The estimated CTR as a fraction (e.g., 0.0035 for 0.35%).
    """
    # Copy the data to avoid modifying the original DataFrame
    data = data.copy()

    # Get the base CTR for the category
    data['base_ctr'] = pd.Series(list(zip(data['main_category'], data['sub_category']))).map(base_ctrs)
    
    # Calculate the rating factor (scale 1-5 stars to a multiplier between 0.8 and 1.2 -> changed to 0.2 to 1.8 -> 0.0 to 1.72)
    #not na rows
    non_nan_mask = data['ratings'].notna()
    data.loc[non_nan_mask, 'rating_factor'] = 1 + 0.5 * (data.loc[non_nan_mask, 'ratings'] - (sum(data.loc[non_nan_mask, 'ratings'])/len(data.loc[non_nan_mask, 'ratings'])))  # Rating 3 is neutral, below 3 decreases CTR, above increases

    nan_mask = data['ratings'].isna()
    data.loc[nan_mask, 'rating_factor'] = 1
    data['rating_factor'] = data['rating_factor'].apply(lambda x: max(x, 0))
    
    # Calculate the discount factor (scale 0-100% discount to a multiplier between 0.9 and 1.5 -> changed to 0.9 to 2.1)  -> 0.8 to2.5 -> 0.5 to 2.2
    non_nan_discount = data['discount_percentage'].notna()
    data.loc[non_nan_discount, 'discount_factor'] = 0.5 + 1.7 * (data.loc[non_nan_discount, 'discount_percentage'].clip(upper=100) / 100)  # Cap discount at 100%

    nan_discount = data['discount_percentage'].isna()
    data.loc[nan_discount, 'discount_factor'] = 0.5

    # price factor (cheaper products in general more likely to be clicked) \in [0.65, 1.65]
    #first everywhere where a discount price is available
    non_nan_price = data['discount_price'].notna()
    nan_aprice = data['discount_price'].isna() & data['actual_price'].isna()
    non_nan_aprice = data['discount_price'].isna() & data['actual_price'].notna()
    data.loc[non_nan_price, 'price_factor'] = 0.65 + np.exp(-(data.loc[non_nan_price, 'discount_price'] / ((sum(data.loc[non_nan_price, 'discount_price'])+sum(data.loc[non_nan_aprice, 'actual_price']))/(len(data.loc[non_nan_price, 'discount_price'])+len(data.loc[non_nan_aprice, 'actual_price'])))))

    # where discount price is not available but the actual price is available
    #nan_price = data['discount_price'].isna()

    data.loc[non_nan_aprice, 'price_factor'] = 0.65 + np.exp(-(data.loc[non_nan_aprice, 'actual_price'] / ((sum(data.loc[non_nan_price, 'discount_price'])+sum(data.loc[non_nan_aprice, 'actual_price']))/(len(data.loc[non_nan_price, 'discount_price'])+len(data.loc[non_nan_aprice, 'actual_price'])))))
    # where both is na set to 0.9
    data.loc[nan_aprice, 'price_factor'] = 0.65

    #cap all between 0 and 10!!
    data['aesthetic_score'] = data['aesthetic_score'].apply(lambda x: max(0, min(x, 10)))
    data['aesthetic_factor'] = 1 + 0.3 * (data['aesthetic_score'] - 5) # rating factor between 0.1 and 1.9 -> 0 and 2.5
    data['aesthetic_factor'] = data['aesthetic_factor'].apply(lambda x: max(x, 0))
    
    # Final CTR calculation
    data['ctr'] = data['base_ctr'] * data['rating_factor'] * data['discount_factor'] * data['aesthetic_factor'] * data['price_factor']

    # add some noise
    data['ctr'] = np.random.normal(data['ctr'], 0.00005)
    
    # Ensure the CTR is within a reasonable range (e.g., 0.001 to 0.05)
    data['ctr'] = data['ctr'].apply(lambda x: max(0.001, min(x, 0.5))) # Cap CTR between 0.1% and 5% -> 50% for now

    return data['ctr']
def apply_gender_split(row):
    """
    Applies gender split based on the heuristic for the given row.
    """
    category_tuple = (row['main_category'], row['sub_category'])  # Adjust column names as needed
    if category_tuple in heuristics:
        male_prob = heuristics[category_tuple]['male_probability']
        # Add noise
        noise = np.random.uniform(-0.025, 0.025)
        male_prob = max(0, min(1, male_prob + noise))  # Ensure within [0, 1]
        female_prob = 1 - male_prob
    else:
        # Default split if no heuristic is found
        male_prob, female_prob = 0.5, 0.5
    return male_prob, female_prob
def apply_age_split(row):
    """
    Applies age group split based on the heuristic for the given row.
    """
    category_tuple = (row['main_category'], row['sub_category'])  # Adjust column names as needed
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-85']
    if category_tuple in heuristics:
        age_probs = heuristics[category_tuple]['age_distribution']
    else:
        # Default uniform split if no heuristic is found
        age_probs = [1 / len(age_groups)] * len(age_groups)

    # Function to generate noisy and normalized probabilities
    def generate_noisy_probs(base_probs, noise_level=0.02):
        noisy_probs = [
            max(0, prob + np.random.uniform(-noise_level, noise_level))
            for prob in base_probs
        ]
        total_prob = sum(noisy_probs)
        return [p / total_prob for p in noisy_probs]
    
    # Generate noisy probabilities for each metric
    sales_probs = generate_noisy_probs(age_probs)
    clicks_probs = generate_noisy_probs(age_probs)
    impressions_probs = generate_noisy_probs(age_probs)
    
    # Initialize dictionary to store metrics
    age_metrics = {}
    for i, age_group in enumerate(age_groups):
        age_metrics[f"sales_{age_group}"] = int(row['estimated_purchases'] * sales_probs[i])
        age_metrics[f"clicks_{age_group}"] = int(row['estimated_clicks'] * clicks_probs[i])
        age_metrics[f"impressions_{age_group}"] = int(row['impressions'] * impressions_probs[i])
        age_metrics[f"sales_{age_group}_male"] = int(row['estimated_purchases_male'] * sales_probs[i])
        age_metrics[f"clicks_{age_group}_male"] = int(row['estimated_clicks_male'] * clicks_probs[i])
        age_metrics[f"impressions_{age_group}_male"] = int(row['impressions_male'] * impressions_probs[i])
        age_metrics[f"sales_{age_group}_female"] = int(row['estimated_purchases_female'] * sales_probs[i])
        age_metrics[f"clicks_{age_group}_female"] = int(row['estimated_clicks_female'] * clicks_probs[i])
        age_metrics[f"impressions_{age_group}_female"] = int(row['impressions_female'] * impressions_probs[i])
        age_metrics[f"percentage_sales_{age_group}"] = sales_probs[i]
        age_metrics[f"percentage_clicks_{age_group}"] = clicks_probs[i]
        age_metrics[f"percentage_impressions_{age_group}"] = impressions_probs[i]
    return age_metrics

def create_data(file_path = "Amazon-Products.csv", out_path = "synthetic_data.parquet"):

    random.seed(6)

    data = pd.read_parquet(file_path)#read_csv(file_path)

    # change prices to float
    #data['actual_price'] = data['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)
    #data['discount_price'] = data['discount_price'].str.replace('₹', '').str.replace(',', '').astype(float)

    # Add a discount percentage column
    #data['discount_percentage'] = ((data['actual_price'] - data['discount_price']) / data['actual_price']) * 100

    # clean number ofratings
    #data['no_of_ratings'] = data['no_of_ratings'].apply(clean_ratings) # pandas converts to float because of containing NaN values

    # clean ratings
    #data['ratings'] = convert_to_float_with_nan(data['ratings'])

    # Correct number of ratings for products where ratings are aggregated across products
    #data['first_word_name'] = data['name'].str.split().str[0] # Extract the first word of the name
    # create groups where first word, rating, no of ratings and category are the same
    #data['group_index'] = data.groupby(['ratings', 'no_of_ratings', 'main_category', 'first_word_name']).ngroup()
    # count group members
    #data['group_count'] = data.groupby('group_index')['group_index'].transform('size')
    # Add the corrected number of ratings (only if group count > number of ratings)
    #data['corrected_no_of_ratings'] = data.apply(
    #    lambda row: int(row['no_of_ratings'] / row['group_count']) 
    #    if row['no_of_ratings'] > row['group_count'] else row['no_of_ratings'], axis=1
    #)

    # estimate purchases based on no of ratings
    #data['estimated_purchases'] = estimate_purchases(data)

    # get the click to purchase ratio and estimate clicks
    #data['clicktopurchase_ratio'] = np.random.normal(data.apply(get_ctp_rate, axis=1), 0.0015)
    #data['estimated_clicks'] = (data['estimated_purchases'] / data['clicktopurchase_ratio']).astype('int64')

    # calculate ctr's based on base-ctr's
    data['ctr'] = calculate_ctr(data, base_ctrs)

    # impressions
    data['impressions'] = (data['estimated_clicks'] / data['ctr']).astype('int64')

    # get gender for sales, clicks and impressions
    data[['male_percentage_purchases', 'female_percentage_purchases']] = data.apply(
        lambda row: pd.Series(apply_gender_split(row)), axis=1
    )
    data[['male_percentage_clicks', 'female_percentage_clicks']] = data.apply(
        lambda row: pd.Series(apply_gender_split(row)), axis=1
    )
    data[['male_percentage_impressions', 'female_percentage_impressions']] = data.apply(
        lambda row: pd.Series(apply_gender_split(row)), axis=1
    )
    data['estimated_purchases_male'] = (data['estimated_purchases'] * data['male_percentage_purchases']).astype('int64')
    data['estimated_purchases_female'] = (data['estimated_purchases'] * data['female_percentage_purchases']).astype('int64')
    data['estimated_clicks_male'] = (data['estimated_clicks'] * data['male_percentage_clicks']).astype('int64')
    data['estimated_clicks_female'] = (data['estimated_clicks'] * data['female_percentage_clicks']).astype('int64')
    data['impressions_male'] = (data['impressions'] * data['male_percentage_impressions']).astype('int64')
    data['impressions_female'] = (data['impressions'] * data['female_percentage_impressions']).astype('int64')

    # Split sales, clicks, impressions (also the already split by gender) into age groups
    age_split_data = data.apply(
        lambda row: pd.Series(apply_age_split(row)), axis=1
    )
    # Merge the new age group columns with the original data
    data = pd.concat([data, age_split_data], axis=1)

    data.to_parquet(out_path, index=False)

    return

if __name__ == '__main__':
    
    create_data()