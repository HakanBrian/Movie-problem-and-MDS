import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt
from itertools import product

# load airport coordinates (OpenFlights CSV: airports.dat)
air = pd.read_csv("https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat", header=None, 
                  names=["id","name","city","country","iata","icao","lat","lon","alt","tz","dst","tzdb","type","source"])
air = air[(air["iata"].notna()) & (air["lat"].notna()) & (air["lon"].notna())]

# define cities and which airports represent them
city_airports = {
    "New York": ["JFK", "LGA", "EWR"],
    "Boston": ["BOS"],
    "Philadelphia": ["PHL"],
    "Washington DC": ["IAD", "DCA", "BWI"],
    "Atlanta": ["ATL"],
    "Miami": ["MIA"],
    "Charlotte": ["CLT"],
    "Nashville": ["BNA"],
    "Tampa": ["TPA"],
    "Chicago": ["ORD", "MDW"],
    "Detroit": ["DTW"],
    "Minneapolis": ["MSP"],
    "St. Louis": ["STL"],
    "Dallas": ["DFW", "DAL"],
    "Houston": ["IAH", "HOU"],
    "Phoenix": ["PHX"],
    "Las Vegas": ["LAS"],
    "Denver": ["DEN"],
    "Salt Lake City": ["SLC"],
    "Los Angeles": ["LAX"],
    "San Francisco": ["SFO", "OAK", "SJC"],
    "Seattle": ["SEA"],
    "Portland": ["PDX"],
    "San Diego": ["SAN"],
    "Anchorage": ["ANC"]
}

# filter to just the airports you need
needed = {a for lst in city_airports.values() for a in lst}
air_subset = air[air["iata"].isin(needed)].set_index("iata")[["lat","lon"]]

# compute pairwise great-circle distance (Haversine)
R = 3958.7613  # Earth radius in miles
def haversine(lat1, lon1, lat2, lon2):
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return 2*R*asin(sqrt(a))

# city-to-city distance: minimum across any airport pair
cities = list(city_airports.keys())
D = pd.DataFrame(np.zeros((len(cities), len(cities))), index=cities, columns=cities)

for c1, c2 in product(cities, cities):
    if c1 == c2:
        D.loc[c1, c2] = 0.0
        continue
    best = np.inf
    for a1 in city_airports[c1]:
        for a2 in city_airports[c2]:
            lat1, lon1 = air_subset.loc[a1, ["lat","lon"]]
            lat2, lon2 = air_subset.loc[a2, ["lat","lon"]]
            best = min(best, haversine(lat1, lon1, lat2, lon2))
    D.loc[c1, c2] = best

# D is the symmetric distance matrix in miles for MDS
D.to_csv("flight_distances.csv")
