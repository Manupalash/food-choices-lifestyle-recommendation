# 🍽️ Food Choices & Location-Based Lifestyle Recommendation

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-K--Means-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Folium](https://img.shields.io/badge/Folium-Geospatial_Map-77B829?style=for-the-badge&logo=leaflet&logoColor=white)](https://python-visualization.github.io/folium/)
[![Foursquare](https://img.shields.io/badge/Foursquare-Places_API-FA4778?style=for-the-badge&logo=foursquare&logoColor=white)](https://location.foursquare.com/developer/)

> **An end-to-end Data Science case study analyzing university student dietary and lifestyle behaviors, clustering population archetypes, and matching them with geographical residential locations based on localized commercial amenities.**

---

## 📌 Table of Contents
- [Executive Summary](#-executive-summary)
- [Project Architecture & Workflow](#-project-architecture--workflow)
- [Dataset & Feature Engineering](#-dataset--feature-engineering)
- [Methodology & Machine Learning](#-methodology--machine-learning)
  - [Part 1: Population Behavioral Clustering](#part-1-population-behavioral-clustering)
  - [Part 2: Geospatial & Amenity Analysis (Foursquare API)](#part-2-geospatial--amenity-analysis-foursquare-api)
  - [Part 3: Population-to-Location Matching Matrix](#part-3-population-to-location-matching-matrix)
- [Interactive Geospatial Visualization](#-interactive-geospatial-visualization)
- [Key Findings & Strategic Takeaways](#-key-findings--strategic-takeaways)
- [Repository Structure](#-repository-structure)
- [Getting Started & Usage](#-getting-started--usage)

---

## 📖 Executive Summary
University students have diverse academic commitments, disposable incomes, fitness routines, and nutritional habits. Finding ideal housing requires balancing personal priorities (home cooking vs. dining out, gym access, proximity to green parks) against neighborhood amenity availability.

This project implements an unsupervised machine learning pipeline using **K-Means Clustering** to:
1. Identify **3 distinct student lifestyle personas** based on academic, socioeconomic, and dietary features.
2. Segment **residential neighborhoods in North Bengaluru** based on nearby amenity density (restaurants, grocery stores, gyms, cafes, and parks within 500m) retrieved via the **Foursquare Places API**.
3. Formulate an evidence-based **Lifestyle-to-Location Recommendation Framework**.

---

## 🔄 Project Architecture & Workflow

```
┌────────────────────────────────────────────────────────┐
│               Part 1: Population Clustering            │
└────────────────────────────────────────────────────────┘
                    Raw Food Choices Dataset
                               │
                               ▼
                         Data Cleaning
             (Type Coercion, Imputation & Dedup)
                               │
                               ▼
                       Feature Selection
                (Academic, Dietary, Lifestyle)
                               │
                               ▼
                   Exploratory Data Analysis
            (Distributions, Boxplots, Correlations)
                               │
                               ▼
                   StandardScaler Normalization
                               │
                               ▼
                 K-Means Population Clustering
            (Elbow Method & Silhouette Validation)
                               │
                               ▼
                Population Profile Interpretation
                               │
                               │
┌──────────────────────────────▼─────────────────────────┐
│          Part 2: Geolocation & Amenity Analysis        │
└────────────────────────────────────────────────────────┘
                   Foursquare Places API
                               │
                               ▼
               Residential Location Discovery
                               │
                               ▼
                Nearby Amenity Aggregation (500m)
            (Restaurants, Groceries, Gyms, Cafes, Parks)
                               │
                               ▼
                    Location-Amenity Dataset
                               │
                               ▼
                  K-Means Location Clustering
                               │
                               ▼
                Interactive Folium Map Generation
                               │
                               │
┌──────────────────────────────▼─────────────────────────┐
│            Synthesis: Recommendation Matrix            │
└────────────────────────────────────────────────────────┘
             Population-to-Location Matching
                               │
                               ▼
                 Key Findings & Actionable Insights
```

---

## 📊 Dataset & Feature Engineering

### Selected Features
| Feature | Category | Scale / Units | Description |
| :--- | :--- | :--- | :--- |
| **`GPA`** | Academic | 0.0 - 4.0 | Academic performance & study dedication |
| **`income`** | Socioeconomic | 1 (Low) to 6 (High) | Financial flexibility & spending power |
| **`weight`** | Physical Health | Pounds (lbs) | Baseline physiological profile |
| **`exercise`** | Lifestyle | 1 (Daily) to 3 (Rarely) | Workout and physical activity frequency |
| **`sports`** | Lifestyle | 1 (Active), 2 (No) | Organized sports participation |
| **`fruit_day`** | Dietary Habit | 1 (Low) to 5 (High) | Daily fruit consumption likelihood |
| **`veggies_day`** | Dietary Habit | 1 (Low) to 5 (High) | Daily vegetable consumption likelihood |
| **`eating_out`** | Food Behavior | 1 (Never) to 5 (Everyday) | Dining out frequency |
| **`cook`** | Food Behavior | 1 (Daily) to 5 (Never) | Home cooking frequency |
| **`pay_meal_out`** | Food Behavior | 1 (<$5) to 6 (>$40) | Expenditure when dining out |

---

## 🤖 Methodology & Machine Learning

### Part 1: Population Behavioral Clustering
- **Feature Standardization:** Normalized using `StandardScaler` to balance multi-scale attributes.
- **Cluster Validation:** Evaluated $K \in [2, 9]$ using the **Elbow Method (Inertia)** and **Silhouette Score Analysis**, confirming **$K=3$** as the optimal cluster count.
- **Empirical Personas:**
  - **Cluster 0 (N = 41) — Academic & Health-Conscious:** Highest GPA (3.49), lowest weight (141.9 lbs), highest fruit (4.54) & veggie (4.59) intake, moderate budget.
  - **Cluster 1 (N = 52) — High-Income & Active Lifestyle:** Highest income (5.31/6), highest sports participation (1.04), regular workouts, balanced diet.
  - **Cluster 2 (N = 32) — Convenience-Driven & Dining-Out:** Lowest fruit (3.13) & veggie (2.63) consumption, higher average weight (170.6 lbs), reliance on dining out.

### Part 2: Geospatial & Amenity Analysis
- **Geographic Center:** Latitude `13.133521`, Longitude `77.567135` (North Bengaluru / Yelahanka University Corridor).
- **Amenity Quantification (500m Radius):** Counts for Restaurants, Grocery Stores, Gyms, Cafes, and Parks.
- **Neighborhood Clusters ($K=3$):**
  - **Location Cluster 0:** *High-Density Fitness & Dining Hub* (Avg. 6.0 restaurants, 2.0 gyms, 1.0 cafe, 1.0 grocery store).
  - **Location Cluster 1:** *Quiet, Green & Park-Oriented Enclave* (Avg. 1.0 park, low commercial dining).
  - **Location Cluster 2:** *Dining & Cafe Urban Corridor* (Avg. 5.0 restaurants, 1.0 cafe, 0.7 gyms).

### Part 3: Population-to-Location Matching Matrix

| Student Population Cohort | Key Lifestyle Drivers | Essential Amenities | Recommended Residential Cluster | Exemplar Complexes |
| :--- | :--- | :--- | :--- | :--- |
| **Cluster 1** *(High-Income & Active)* | Fitness routine, high disposable income, diverse food preferences | Gyms, cafes, restaurants, grocery markets | **Location Cluster 0** *(Amenity-Rich Hub)* | *Shriram Suhaana Apartments*, *shobha azalia* |
| **Cluster 0** *(Academic & Health-Conscious)* | High study focus, home cooking, outdoor wellness, moderate budget | Green parks, fresh groceries, quiet study areas | **Location Cluster 1** *(Quiet & Park Enclave)* | *Prestige Royal Garden*, *bmsit boys hostel*, *Prestige Garden Bay* |
| **Cluster 2** *(Convenience-Driven)* | Frequent dining out, quick coffee access, convenience orientation | High restaurant density, coffee shops, transit corridors | **Location Cluster 2** *(Dining Corridor)* | *GK Lakeview Apartment*, *NCC Urban - Nagarjuna Meadows* |

---

## 🗺️ Interactive Geospatial Visualization
The project exports a fully interactive **Folium map** featuring:
- Custom color markers representing neighborhood clusters.
- Rich HTML popups with building names and itemized amenity breakdowns.
- Floating legend and mini-map navigation.

The map is saved at [`outputs/maps/clustered_locations_map.html`](outputs/maps/clustered_locations_map.html).

---

## 📁 Repository Structure
```
gelocation_data_notebook/
├── Geolocational_data.ipynb         # Main portfolio-ready Jupyter Notebook
├── food_coded.csv                  # Student dietary & lifestyle survey dataset
├── location_amenities.csv          # Extracted Foursquare amenity counts per location
├── codebook_food.docx              # Survey codebook & variable descriptions
├── outputs/
│   └── maps/
│       └── clustered_locations_map.html  # Interactive Folium clustered map
├── README.md                       # Comprehensive project documentation
└── .gitignore                      # Git ignore file for Python/Jupyter artifacts
```

---

## 🚀 Getting Started & Usage

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd gelocation_data_notebook
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn folium requests
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook Geolocational_data.ipynb
```

*(Optional)* To query live Foursquare data rather than cached results:
```bash
export FOURSQUARE_API_KEY="your_api_key_here"
```

---

## 👤 Author
- **Manu Palash** - Data Science & Machine Learning Portfolio
